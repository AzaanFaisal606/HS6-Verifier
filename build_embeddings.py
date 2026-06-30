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
DB = HERE / "pct_corpus.db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
BATCH = 256


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def pack(vec) -> bytes:
    """Serialize a 384-float vector to sqlite-vec's compact float32 blob."""
    return struct.pack(f"{DIM}f", *vec)


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


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
      - national  : segments[2:]    (drop chapter + heading; keeps the
                    subheading-grouping labels that distinguish HS6 — e.g.
                    'Articles ... carried in the pocket:' vs 'Other:' — PLUS the
                    leaf's own label. This is the '6-digit sub + 7th/8th local
                    classified together' text.)
      - subheading: not a query stage — skipped.

    Slicing description_full (not re-joining raws) is what preserves the
    discriminating grouping label, which lives only in the chain, never in the
    subheading's description_raw.

    Returns {pct_code: own_text} for chapter/heading/national rows only.
    """
    cut = {"chapter": 0, "heading": 1, "national": 2}
    out: dict[str, str] = {}
    for code, level, dfull in con.execute(
        "SELECT pct_code, level, description_full FROM codes"
    ):
        if level not in cut or not dfull:
            continue  # subheading rows skipped on purpose
        segs = dfull.split(" > ")
        text = " > ".join(segs[cut[level]:]).strip()
        if text:
            out[code] = text
    return out


def build(db_path: Path):
    con = connect(db_path)
    con.execute("DROP TABLE IF EXISTS vec_embeddings")
    con.execute(
        """
        CREATE VIRTUAL TABLE vec_embeddings USING vec0(
            embedding         FLOAT[384] distance_metric=cosine,
            embedding_own     FLOAT[384] distance_metric=cosine,
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

    model = load_model()
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
          chapter: str | None = None):
    con = connect(db_path)
    model = load_model()
    qvec = embed_texts(model, [text])[0]

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
    args = ap.parse_args()

    if args.query:
        query(Path(args.db), args.query, args.k, args.level, args.chapter)
    else:
        build(Path(args.db))


if __name__ == "__main__":
    main()
