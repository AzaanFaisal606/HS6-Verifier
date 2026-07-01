#!/usr/bin/env python3
"""Embed every `codes.description_full` into a sqlite-vec virtual table.

Builds the `vec_embeddings` vec0 table inside the SAME pct_corpus.db, next to the
relational `codes` table. The corpus is small (~14k rows) so a brute-force vec0
table is plenty.

sqlite-vec virtual tables cannot declare a foreign key, so `pct_code` is carried
as a filterable metadata column and used as a logical FK back to codes.pct_code
for joins. `level` and `chapter` are also stored as filterable metadata so KNN
queries can be pre-filtered cheaply. `description_full` is an auxiliary (+) column
— a side-car copy for quick debug display without a join.

Embedding model: BAAI/bge-small-en-v1.5 (384-dim) via sentence-transformers.
Distance metric: cosine (vectors are L2-normalized, so cosine == dot).

Run:  python3 build_embeddings.py            # embed all rows
      python3 build_embeddings.py --query "wiring harness for cars"
"""

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec

HERE = Path(__file__).resolve().parent
# Universal embedder: works against any *_corpus.db with a `codes` table.
# Default points at the US HST corpus; pass --db pct/pct_corpus.db for PCT.
DB = HERE / "hst" / "hst_corpus.db"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DIM = 1024
# Qwen3-Embedding-0.6B on CPU is heavy. A large batch (was 256) padded to the
# longest sequence builds multi-GB transient activations that spike host RAM and
# crash the WSL2 VM (not a Linux OOM — the VM dies before the kernel logs it).
# Keep this small for CPU runs.
BATCH = 16

# Qwen3-Embedding is asymmetric: QUERIES are wrapped with a task instruction,
# DOCUMENTS (the corpus side built here) are embedded RAW. The query-side wrap
# lives in the inference clients (test_infer*). Must stay in sync with them.


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def pack(vec) -> bytes:
    """Serialize a DIM-float vector to sqlite-vec's compact float32 blob."""
    return struct.pack(f"{DIM}f", *vec)


def load_model(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    if device == "cpu":
        # Don't let the embedder grab every core; a fully-pinned host CPU
        # contributes to the WSL2 VM stall/crash. Leave headroom for the host.
        import torch
        torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))
    return SentenceTransformer(MODEL_NAME, device=device)


# Qwen3-Embedding query-side instruction. Documents (corpus) are embedded raw;
# only the query is wrapped. Keep identical to the inference clients.
QUERY_INSTRUCT = (
    "Instruct: Given a product description, retrieve the matching "
    "Harmonized System (HS) tariff classification.\nQuery: "
)


def query_text(text: str) -> str:
    return f"{QUERY_INSTRUCT}{text}"


def embed_texts(model, texts: list[str]):
    # normalize so cosine distance is well-behaved; bge models recommend it
    return model.encode(
        texts, batch_size=BATCH, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )


def own_description(con: sqlite3.Connection) -> dict[str, str]:
    """Per-code 'own' text for the hierarchical (dense-emb) retrieval variant,
    TOP PARENTS DISCARDED so same-level rows are not near-identical.

    description_full is the chain  chapter > heading > [grouping labels...] > own.
    The cascade classifies one level at a time, so each level's vector should NOT
    carry the parents already fixed by an earlier stage:

      - chapter   : segment[0]      (its own title).
      - heading   : segments[1:]    (drop chapter; heading desc + any below).
      - national  : segments[2:]    (PCT 8-digit leaf — drop chapter + heading;
                    keeps the subheading-grouping labels that distinguish HS6 —
                    e.g. 'Articles ... carried in the pocket:' vs 'Other:' — PLUS
                    the leaf's own label.)
      - subheading: segments[2:]    (HST HS6 leaf — same cut as national; this is
                    the leaf the HST cascade ends on.)

    Slicing description_full (not re-joining raws) is what preserves the
    discriminating grouping label, which lives only in the chain, never in the
    subheading's description_raw.

    Returns {pct_code: own_text} for chapter/heading/leaf (national|subheading)
    rows.
    """
    # national (PCT 8-digit leaf) and subheading (HST HS6 leaf) both cut=2:
    # drop chapter+heading, keep the discriminating grouping labels + own label.
    cut = {"chapter": 0, "heading": 1, "national": 2, "subheading": 2}
    out: dict[str, str] = {}
    for code, level, dfull in con.execute(
        "SELECT pct_code, level, description_full FROM codes"
    ):
        if level not in cut or not dfull:
            continue
        segs = dfull.split(" > ")
        text = " > ".join(segs[cut[level]:]).strip()
        if text:
            out[code] = text
    return out


def build(db_path: Path, device: str = "cpu"):
    con = connect(db_path)
    con.execute("DROP TABLE IF EXISTS vec_embeddings")
    con.execute(
        f"""
        CREATE VIRTUAL TABLE vec_embeddings USING vec0(
            embedding         FLOAT[{DIM}] distance_metric=cosine,
            embedding_own     FLOAT[{DIM}] distance_metric=cosine,
            pct_code          TEXT,
            level             TEXT,
            chapter           TEXT,
            +description_full TEXT,
            +description_own  TEXT
        )
        """
    )

    rows = con.execute(
        "SELECT pct_code, level, description_full FROM codes "
        "WHERE description_full IS NOT NULL AND description_full != '' "
        "ORDER BY pct_code"
    ).fetchall()
    print(f"rows to embed: {len(rows)}")

    own = own_description(con)  # {pct_code: own_text} for chapter/heading/national

    model = load_model(device)
    full_texts = [r[2] for r in rows]
    # own text falls back to the full chain when a row has no own text (e.g.
    # subheading rows, or empty-raw rows) so every row still carries an
    # embedding_own vector — the cascade just never queries the skipped levels.
    own_texts = [own.get(r[0]) or r[2] for r in rows]

    print("embedding description_full ...")
    full_vecs = embed_texts(model, full_texts)
    print("embedding description_own ...")
    own_vecs = embed_texts(model, own_texts)

    payload = []
    for (pct_code, level, desc_full), fvec, ovec, otext in zip(
        rows, full_vecs, own_vecs, own_texts
    ):
        chapter = pct_code[:2]
        payload.append(
            (pack(fvec.tolist()), pack(ovec.tolist()),
             pct_code, level, chapter, desc_full, otext)
        )

    con.executemany(
        "INSERT INTO vec_embeddings"
        "(embedding, embedding_own, pct_code, level, chapter, "
        " description_full, description_own) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        payload,
    )
    con.commit()

    n = con.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()[0]
    print(f"vec_embeddings rows: {n}")
    con.close()


def query(db_path: Path, text: str, k: int = 8, level: str | None = None,
          chapter: str | None = None, device: str = "cpu"):
    con = connect(db_path)
    model = load_model(device)
    # query side gets the Qwen3-Embedding task instruction; corpus side is raw.
    qvec = embed_texts(model, [query_text(text)])[0]

    where = ["embedding MATCH ?", "k = ?"]
    params: list = [pack(qvec.tolist()), k]
    # metadata pre-filters (optional)
    if level:
        where.append("level = ?")
        params.append(level)
    if chapter:
        where.append("chapter = ?")
        params.append(chapter)
    sql = (
        "SELECT pct_code, level, chapter, distance, description_full "
        "FROM vec_embeddings WHERE " + " AND ".join(where) + " ORDER BY distance"
    )
    print(f"\nquery: {text!r}\n" + "-" * 70)
    for pct_code, lvl, ch, dist, desc in con.execute(sql, params):
        print(f"{dist:.4f}  {pct_code:11s} [{lvl:10s}] {desc[:90]}")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--query", help="run a semantic search instead of building")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--level", help="filter results to a level")
    ap.add_argument("--chapter", help="filter results to a 2-digit chapter")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the embedder: cpu (default, safe to "
                         "run alongside the vLLM server) or cuda (faster; only "
                         "when the GPU is free, e.g. VLM stopped).")
    args = ap.parse_args()

    if args.query:
        query(Path(args.db), args.query, args.k, args.level, args.chapter,
              args.device)
    else:
        build(Path(args.db), args.device)


if __name__ == "__main__":
    main()
