"""Single-call VLM product extraction for tariff classification.

One request: the model looks at the image, reasons briefly about the product,
and emits a JSON object matching the schema. Output (any reasoning + the JSON)
is printed and the JSON is parsed/validated.

Why single-call (not a 2-phase analyze->structure split): this is the Qwen3-VL
*Thinking* variant. On a constrained text-only "structure this" step it reasons
without terminating and burns the whole token budget before emitting JSON. But
image -> brief reasoning -> JSON in ONE call terminates cleanly (finish=stop)
and fits the 3072-token window with room to spare. So we keep it to one call.

Usage:
    python test_infer.py [image_path_or_url]
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
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384

# --- Hierarchical (dense-emb) cascade knobs -------------------------------
# Query drills down one HS level at a time against the `embedding_own` vectors
# (each row embedded on its OWN text, top parents discarded — see
# build_embeddings.own_description). Stages: chapter -> heading -> subheading.
# The HST corpus stops at HS6, so the subheading IS the leaf/final stage; its
# own text carries the discriminating 6-digit grouping label + own label.
N_CHAP = 2   # chapters kept after stage 1 (beam width — drill into the best N)
N_HEAD = 6   # headings kept after stage 2 (beam width — a generic 'Other ...'
             # heading can out-score the right enumerated heading on own-desc, so
             # keep several; stage 3 KNNs subheadings across ALL kept headings and
             # the strong leaf match wins. Raise if the right code is pruned.
             # 6 (not 3): HST's ~1262 headings crowd stage 2 — at 3 the right
             # heading (e.g. 4202 for a pocket wallet) gets pruned; at 6 it
             # survives and 4202.31 returns rank #1.
K_NAT  = 5   # subheading (HS6) codes returned by stage 3 (final KNN k)

# Keep this prompt SHORT. A long, constraint-heavy prompt makes the Thinking
# model reason without terminating (burns the whole token budget, emits no JSON).
# A terse "look, brief reason, output JSON" reliably finishes (finish=stop).
PROMPT = """You extract product attributes for customs tariff (HS) classification. Think
in 3 short sentences max, then output JSON. Do not exceed that — brevity is required.

RULES:
- Describe ONLY the product. NEVER mention background, surface it rests on,
  lighting, or photo setting.
- State what the object IS (its common article name: e.g. wallet, purse,
  belt, key-case) and HOW/WHERE a person carries or uses it (pocket, handbag,
  worn, desk) — these carry/use facts are critical for classification.
- Report only visually-verifiable construction and material. Do NOT guess
  fiber content, hide/skin species, or genuine-vs-synthetic — put those in
  uncertain_attributes.
- embedding_description must be ONE tariff-style sentence that leads with the
  article name and its carry/use context, then outer-surface material, then
  construction. No colours-only, no scene, no filler.

Output ONLY this JSON (nothing after):
{
  "category": "<common article name>",
  "function": "<how/where carried or used>",
  "visible_construction": "<stitching/closure/fold — only if evident>",
  "visible_materials": ["<outer surface material as it appears>"],
  "embedding_description": "<article name + carry/use context + outer-surface
     material + construction, one sentence, tariff register>",
  "uncertain_attributes": ["<attr>: <why not determinable from image>"],
  "confidence_notes": "<brief>"
}"""


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


_embed_model = None


def embed(text: str):
    """Embed one string with the corpus model (L2-normalized, cosine-ready)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    )[0]


def _connect():
    con = sqlite3.connect(DB)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def _knn_own(con, blob, level: str, k: int, prefixes=None, chapter=None):
    """KNN against the `embedding_own` vectors, restricted to one HS level and
    (optionally) to descendants of a winning parent.

    vec0 KNN only allows EQUALITY/inequality on metadata columns inside a MATCH
    query — no LIKE, no OR, no IN. So we can pre-filter cheaply on `level` and
    `chapter` (both metadata `=`), but a pct_code PREFIX narrowing (e.g. "only
    nationals under heading 42.02") is NOT expressible in the MATCH. For that we
    over-fetch a generous KNN and filter the prefixes in Python afterward.

      - chapter  : optional 2-digit, applied as a legal `chapter = ?` pre-filter.
      - prefixes : optional list of pct_code prefixes (e.g. ['4202']); applied in
                   Python post-KNN. We over-fetch (k * fanout) so enough in-prefix
                   rows survive the filter before we slice to k.

    Returns rows (pct_code, distance, description_own) ordered by distance.
    """
    fetch = k if prefixes is None else max(k * 40, 200)  # over-fetch for py filter
    sql = ("SELECT pct_code, distance, description_own "
           "FROM vec_embeddings WHERE embedding_own MATCH ? AND k = ? AND level = ?")
    params = [blob, fetch, level]
    if chapter is not None:
        sql += " AND chapter = ?"
        params.append(chapter)
    sql += " ORDER BY distance"
    rows = con.execute(sql, params).fetchall()
    if prefixes is not None:
        rows = [r for r in rows if any(r[0].startswith(p) for p in prefixes)]
    return rows[:k]


def retrieve(embedding_description: str, n_chap: int = N_CHAP,
             n_head: int = N_HEAD, k_nat: int = K_NAT):
    """Hierarchical drill-down: chapter -> heading -> subheading, all scored on
    the `embedding_own` vectors (own text, top parents discarded). No DB writes.

    Stage 1  KNN chapters                        -> keep best n_chap chapter codes
    Stage 2  KNN headings within those chapters  -> keep best n_head heading codes
    Stage 3  KNN subheadings within those heads  -> return top k_nat (final answer,
             the HS6 leaf)
    """
    con = _connect()
    blob = struct.pack(f"{DIM}f", *embed(embedding_description).tolist())

    print(f"\n{'='*72}\nHIERARCHICAL CASCADE (embedding_own, cosine)\n{'='*72}")
    print(f"query: {embedding_description!r}")
    print(f"knobs: N_CHAP={n_chap} N_HEAD={n_head} K_NAT={k_nat}\n")

    # Stage 1 — chapters (own = chapter title). KNN over all chapter rows.
    chap = _knn_own(con, blob, "chapter", n_chap)
    print("-- stage 1: chapters --")
    for code, dist, own in chap:
        print(f"   sim {1-dist:.4f}  ch {code}  {own[:70]}")
    chap_codes = [c for c, _, _ in chap]
    if not chap_codes:
        print("!! no chapter hit"); con.close(); return

    # Stage 2 — headings within the winning chapter(s). vec0 KNN takes one
    # `chapter =` value, so query per chapter and merge by distance.
    head = []
    for ch in chap_codes:
        head += _knn_own(con, blob, "heading", n_head, chapter=ch)
    head.sort(key=lambda r: r[1])
    head = head[:n_head]
    print("\n-- stage 2: headings --")
    for code, dist, own in head:
        print(f"   sim {1-dist:.4f}  {code:8s} {own[:66]}")
    head_codes = [c for c, _, _ in head]
    if not head_codes:
        print("!! no heading hit"); con.close(); return

    # heading pct_code is "NNNN" (4-digit, no dot); subheading codes are
    # "NNNN.XX" — strip the dot so the prefix filter matches the HS6 leaves
    # under the heading (e.g. heading "4202" -> "4202" matches "420221").
    head_prefixes = [c.replace(".", "") for c in head_codes]
    # subheading rows all sit in the heading's chapter — pass it as a legal vec0
    # pre-filter to shrink the pool before the Python prefix filter.
    nat_chapter = head_codes[0][:2] if len(chap_codes) == 1 else None

    # Stage 3 — subheadings within the winning heading(s) = the final answer.
    nat = _knn_own(con, blob, "subheading", k_nat,
                   prefixes=head_prefixes, chapter=nat_chapter)
    print(f"\n-- stage 3: TOP {k_nat} SUBHEADING (HS6) CODES --")
    for code, dist, own in nat:
        full = con.execute(
            "SELECT description_full FROM vec_embeddings WHERE pct_code = ?",
            (code,),
        ).fetchone()[0]
        print(f"   sim {1-dist:.4f}  {code:11s}")
        print(f"               own : {own}")
        print(f"               full: {full}\n")
    con.close()


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_image_url(image)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=2100,
        temperature=0.3,
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = (msg.content or "").strip()

    print(f"{'='*72}\nMODEL OUTPUT\n{'='*72}")
    if reasoning:
        print("--- thinking ---")
        print(reasoning.strip())
        print("--- output ---")
    print(content)
    print(f"\n[usage] {resp.usage}  finish={resp.choices[0].finish_reason}")

    print(f"\n{'='*72}\nPARSED JSON\n{'='*72}")
    try:
        parsed = json.loads(extract_json(content))
    except json.JSONDecodeError as e:
        print(f"!! JSON parse failed: {e}")
        return
    print(json.dumps(parsed, indent=2))

    # embed the embedding_description and retrieve nearest corpus codes (no save)
    desc = (parsed.get("embedding_description") or "").strip()
    if desc:
        retrieve(desc)
    else:
        print("\n!! no embedding_description in output — skipping retrieval")


if __name__ == "__main__":
    main()
