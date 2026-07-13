"""Chapter-gated VLM product classification for tariff (HS) codes.

Two classification stages, ONE model call:

  Stage A (in the single VLM call): the model is given the FULL catalog of HS
  chapter titles and, alongside the usual product attributes, emits the top-3
  chapters it thinks the product belongs to.
  Stage B (retrieval): flat KNN on the corpus vector column VEC_COLUMN, pre-
  filtered to level='subheading' AND restricted to the chapters the model chose,
  then reranked with the cross-encoder. This is the same retrieve -> rerank
  pipeline as test_infer.py — the only difference is the chapter gate replaces
  the wide all-chapter pool (and there is no reinfer step here).

The corpus register is swappable via VEC_COLUMN/DOC_COLUMN (default: the full
ancestor chain, as in matrix Runs 7/8; `embedding_ai`/`description_ai` scores the
register-normalized rewrite instead). Both the KNN and the reranker read it.

Why chapter-gate instead of the old level-by-level cascade: the previous dense
strategy drilled chapter->heading->subheading on `embedding_own`, and a wrong
prune at any stage was unrecoverable. Here the model picks the chapter set up
front (it is good at coarse "what kind of thing is this"), and the leaf pool
inside those chapters stays wholly alive for the reranker.

Usage:
    python test_infer_dense.py [image_path_or_url]
"""
import base64
import json
import mimetypes
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "qwen3-vl"

DEFAULT_IMG = "test.jpeg"

# Corpus retrieval — must match build_embeddings.py (same model/dim/metric).
DB = Path(__file__).resolve().parent / "hst" / "hst_corpus.db"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DIM = 1024
TOP_K = 50          # cosine pool size handed to the reranker (across the gate)
SHOW_N = 5
N_CHAPTERS = 3      # how many chapters the model outputs / we gate retrieval to
# Which corpus register each stage scores against. Registers pair as:
#   embedding      / description_full  — full ancestor chain (legalese)
#   embedding_own  / description_own   — chain with top parents sliced off
#   embedding_ai   / description_ai    — register-normalized rewrite (HS6 only)
#
# VEC_COLUMN picks the vectors the cosine KNN searches; DOC_COLUMN is the text
# it returns for display. RERANK_DOC_COLUMN is what the cross-encoder actually
# judges, and it is INDEPENDENT — the two stages want different registers (the
# bi-encoder retrieves better on the dense rewrite; the cross-encoder needs the
# ancestor chain to separate near-identical siblings like 6505.00 hat vs 6507.00
# headband). Set it to None to reuse DOC_COLUMN (the paired default, Runs 7/8).
VEC_COLUMN = "embedding"
DOC_COLUMN = "description_full"
RERANK_DOC_COLUMN = None
# CPU embedder/reranker: coexist with the ~11GB vLLM server on one 12GB GPU.
EMBED_DEVICE = "cpu"
RERANK_DEVICE = "cpu"
# Qwen3-Embedding asymmetric: query gets the instruction, corpus is raw.
# Must match build_embeddings.QUERY_INSTRUCT.
QUERY_INSTRUCT = (
    "Instruct: Given a product description, retrieve the matching "
    "Harmonized System (HS) tariff classification.\nQuery: "
)
RERANK_INSTRUCT = (
    "Given a product description, decide if the candidate tariff (HS) "
    "classification describes the product."
)

# Prompt carries the full chapter catalog ({chapter_catalog}) so the model can
# choose from the SAME chapters that exist in the corpus. JSON braces are doubled
# for str.format.
PROMPT = """You extract product attributes for customs tariff (HS) classification for US imports. We are using the HST schedule for codes and descriptions.

Here is the full list of HS chapters (2-digit code + title):
{chapter_catalog}

RULES:
- Describe ONLY the product. NEVER mention background, surface it rests on,
  lighting, or photo setting.
- State what general category it comes under (e.g. Bovine animal, Data processing machine, Organic chemical).
- State what the object IS (its common article name: e.g. wallet, purse,
  belt, key-case).
- HOW it is used / HOW it functions.
- If the item is held on person, then WHERE a person carries or uses it (pocket, handbag,
  worn, desk).
- Report only visually-verifiable construction and material. Do NOT guess
  fiber content, hide/skin species, or genuine-vs-synthetic — put those in
  uncertain_attributes. Also only report material makeup if the material of the object is relevant to its functionality.
- embedding_description must be ONE tariff-style sentence that leads with the
  article name and its category, carry/use context, then outer-surface material, then
  construction. No colours-only, no scene, no filler.
- Use tarrif terminology for categorizing the product.
- chapters: pick the {n_chapters} chapters from the list above that are MOST
  likely to contain this product, ordered most-likely first. Output their
  2-digit codes exactly as shown (e.g. "42"). Choose ONLY from the list.

Output ONLY this JSON (nothing after):
{{
  "name": "<common article name>",  
  "category": "<general article category>",
  "function": "<how/where carried or used>",
  "visible_construction": "<stitching/closure/fold — only if evident>",
  "visible_materials": ["<outer surface material as it appears>"],
  "chapters": ["<2-digit>", "<2-digit>", "<2-digit>"],
  "embedding_description": "<article name + carry/use context + outer-surface
     material + construction, one sentence, tariff register>",
  "uncertain_attributes": ["<attr>: <why not determinable from image>"],
  "confidence_notes": "<brief>"
}}"""


def to_image_url(src: str) -> str:
    if src.startswith(("http://", "https://")):
        return src
    mime = mimetypes.guess_type(src)[0] or "image/jpeg"
    with open(src, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> str:
    """Pull the last balanced {...} object out of a text blob."""
    if not text:
        return text
    end = text.rfind("}")
    if end == -1:
        return text
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return text[i:end + 1]
    return text


def load_chapters():
    """Return [(code, title)] for every chapter row, ordered by code.

    Plain sqlite (no vec extension needed) — reads the `codes` table.
    """
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT pct_code, description_raw FROM codes "
        "WHERE level='chapter' ORDER BY pct_code"
    ).fetchall()
    con.close()
    return rows


_embed_model = None


def embed(text: str):
    """Embed one QUERY string (Qwen3-Embedding instruction-wrapped, CPU)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embed_model.encode(
        [f"{QUERY_INSTRUCT}{text}"], normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


# vector column -> the aux text column it was built from. Whitelist: these names
# are interpolated into SQL (vec0 MATCH takes a column, not a bind param), so a
# caller can only ever select one of these known-good pairs.
VEC_DOC_PAIRS = {
    "embedding": "description_full",
    "embedding_own": "description_own",
    "embedding_ai": "description_ai",
}


def retrieve(embedding_description: str, chapters: list, k: int = TOP_K,
             level: str = "subheading", verbose: bool = True):
    """Flat KNN on the VEC_COLUMN vector, pre-filtered to `level` and GATED to
    `chapters`.

    vec0 KNN allows only `=`/`!=`/`<`/`>` on metadata — no IN/OR. So we KNN once
    per chapter with a legal `chapter = ?` filter, merge the pools, and take the
    globally-nearest k across the gate. Returns (pct_code, level, distance,
    rerank_doc) 4-tuples — the shape rerank() expects. rerank_doc is
    RERANK_DOC_COLUMN when set, else DOC_COLUMN, so the retrieval register and
    the rerank register can differ (retrieve on the dense rewrite, judge on the
    full chain).
    """
    if VEC_COLUMN not in VEC_DOC_PAIRS:
        raise ValueError(f"unknown VEC_COLUMN {VEC_COLUMN!r}; "
                         f"valid: {', '.join(VEC_DOC_PAIRS)}")
    rr_doc = RERANK_DOC_COLUMN or DOC_COLUMN
    for name, col in (("DOC_COLUMN", DOC_COLUMN),
                      ("RERANK_DOC_COLUMN", rr_doc)):
        if col not in VEC_DOC_PAIRS.values():
            raise ValueError(f"unknown {name} {col!r}; valid: "
                             f"{', '.join(VEC_DOC_PAIRS.values())}")

    con = sqlite3.connect(DB)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)

    qvec = embed(embedding_description)
    blob = struct.pack(f"{DIM}f", *qvec.tolist())

    sql = (f"SELECT pct_code, level, distance, {rr_doc} "
           f"FROM vec_embeddings "
           f"WHERE {VEC_COLUMN} MATCH ? AND k = ? AND chapter = ?")
    if level:
        sql += " AND level = ?"
    sql += " ORDER BY distance"

    rows = []
    for ch in chapters:
        params = [blob, k, ch] + ([level] if level else [])
        rows += con.execute(sql, params).fetchall()
    con.close()

    rows.sort(key=lambda r: r[2])  # global nearest across the gated chapters
    rows = rows[:k]

    if verbose:
        tag = f" [level={level}]" if level else ""
        print(f"\n{'='*72}\nRETRIEVE: top {SHOW_N} of {len(rows)} cosine candidates"
              f"{tag}  gate=chapters {chapters}  knn={VEC_COLUMN} "
              f"rerank_doc={rr_doc}\n{'='*72}")
        print(f"query: {embedding_description!r}\n")
        for pct_code, lvl, dist, doc in rows[:SHOW_N]:
            print(f"sim {1 - dist:.4f}  {pct_code:11s} [{lvl}]")
            print(f"            {doc}\n")
    return rows


_rerank_model = None


def rerank(embedding_description: str, candidates: list, show_n: int = SHOW_N,
           verbose: bool = True):
    global _rerank_model
    if not candidates:
        if verbose:
            print("\n!! no candidates to rerank")
        return []
    if _rerank_model is None:
        import torch
        from sentence_transformers import CrossEncoder
        if RERANK_DEVICE == "cpu":
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))
        _rerank_model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)

    docs = [c[3] for c in candidates]
    pairs = [(embedding_description, d) for d in docs]
    scores = _rerank_model.predict(pairs, prompt=RERANK_INSTRUCT)

    ranked = sorted(
        ((float(s), c[0], c[1], c[3]) for s, c in zip(scores, candidates)),
        key=lambda x: -x[0],
    )

    if verbose:
        print(f"\n{'='*72}\nRERANK: top {show_n} of {len(candidates)} "
              f"(Qwen3-Reranker-0.6B)\n{'='*72}")
        print(f"query: {embedding_description!r}\n")
        for score, pct_code, lvl, desc_full in ranked[:show_n]:
            print(f"score {score:+.4f}  {pct_code:11s} [{lvl}]")
            print(f"            {desc_full}\n")
    return ranked


def norm6(code) -> str:
    """Normalize an HS code to NNNN.NN (6-digit subheading form)."""
    digits = "".join(ch for ch in str(code) if ch.isdigit())[:6]
    return f"{digits[:4]}.{digits[4:6]}" if len(digits) >= 6 else str(code)


def classify(image: str, verbose: bool = False) -> dict:
    """Run the chapter-gated KNN + rerank flow (dense pipeline).

    Same logic and params as the CLI. Prints section blocks only when
    verbose=True; always returns the standard result dict:

        {flow, hs6, description, query, caption, chapters, candidates, meta}

    `chapters` carries the gate the model picked. candidates:
    [{hs6, description, score, score_type, level}, ...] (top SHOW_N). On any
    failure returns a dict with error set and hs6=None.
    """
    image_url = to_image_url(image)

    chapters = load_chapters()
    valid_codes = {c for c, _ in chapters}
    catalog = "\n".join(f"{c} {title}" for c, title in chapters)
    prompt = PROMPT.format(chapter_catalog=catalog, n_chapters=N_CHAPTERS)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=2000,
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = (msg.content or "").strip()

    if verbose:
        print(f"{'='*72}\nMODEL OUTPUT\n{'='*72}")
        if reasoning:
            print("--- thinking ---")
            print(reasoning.strip())
            print("--- output ---")
        print(content)
        print(f"\n[usage] {resp.usage}  finish={resp.choices[0].finish_reason}")

    def err(msg_, picked_=None, query_=None, parsed_=None):
        if verbose:
            print(f"\n!! {msg_}")
        return {"flow": "dense", "hs6": None, "description": None,
                "query": query_, "caption": parsed_,
                "chapters": picked_ or [], "candidates": [], "meta": {},
                "error": msg_}

    try:
        parsed = json.loads(extract_json(content))
    except json.JSONDecodeError as e:
        return err(f"JSON parse failed: {e}")
    if verbose:
        print(f"\n{'='*72}\nPARSED JSON\n{'='*72}")
        print(json.dumps(parsed, indent=2))

    base_desc = (parsed.get("embedding_description") or "").strip()
    if not base_desc:
        return err("no embedding_description in output", parsed_=parsed)

    # Composed query (small_changes.md 2026-07-07): append function + category
    # to embedding_description so the discriminator (often placed in `function`,
    # Issue 1) and the coarse category actually reach the embedded text.
    func = (parsed.get("function") or "").strip()
    cat = (parsed.get("category") or "").strip()
    parts = [base_desc]
    if func:
        parts.append(f"Function: {func}.")
    if cat:
        parts.append(f"Category: {cat}.")
    desc = " ".join(parts)

    # Normalize the model's chapter picks: 2-digit, valid, deduped, order kept.
    picked, seen = [], set()
    for c in parsed.get("chapters", []):
        c = str(c).strip().zfill(2)
        if c in valid_codes and c not in seen:
            seen.add(c)
            picked.append(c)
    picked = picked[:N_CHAPTERS]

    if not picked:
        return err("model returned no valid chapters — cannot gate retrieval",
                   query_=desc, parsed_=parsed)

    candidates = retrieve(desc, picked, verbose=verbose)
    ranked = rerank(desc, candidates, verbose=verbose)

    if not ranked:
        return err("empty rerank pool", picked_=picked, query_=desc,
                   parsed_=parsed)

    cands = [{"hs6": norm6(pc), "description": dfull, "score": sc,
              "score_type": "rerank", "level": lvl}
             for sc, pc, lvl, dfull in ranked[:SHOW_N]]
    top = cands[0]
    return {
        "flow": "dense",
        "hs6": top["hs6"],
        "description": top["description"],
        "query": desc,
        "caption": parsed,
        "chapters": picked,
        "candidates": cands,
        "meta": {},
    }


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    classify(image, verbose=True)


if __name__ == "__main__":
    main()
