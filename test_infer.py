
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
# Retrieve a WIDE cosine pool (TOP_K) but only DISPLAY the best few (SHOW_N).
# The full pool is fed to the reranker, which then re-orders it. K is wide on
# purpose: cosine smears material/negation (e.g. "not leather" ~0.84 to
# "leather"), so the true code can sit deep in the pool — the reranker promotes
# it, but only if it's IN the pool. Don't shrink K without reason.
TOP_K = 50
SHOW_N = 5
# Embedder runs on CPU: this client coexists with the ~11GB vLLM server on the
# same 12GB GPU, so a GPU embedder would OOM. 0.6B on CPU for one query is ~1-2s.
EMBED_DEVICE = "cpu"
# Qwen3-Embedding is asymmetric — the QUERY gets a task instruction, the corpus
# (built in build_embeddings.py) is raw. Must match build_embeddings.QUERY_INSTRUCT.
QUERY_INSTRUCT = (
    "Instruct: Given a product description, retrieve the matching "
    "Harmonized System (HS) tariff classification.\nQuery: "
)

# Qwen3-Reranker is an LLM judge (Qwen3ForCausalLM + a yes/no logit head): for a
# (query, document) pair it scores P("yes") that the document answers the query.
# It needs a TASK INSTRUCTION describing what to judge — without it the model
# behaves like a dumb similarity scorer (it will keep "leather" on top for a
# "not leather" query). The instruction is deliberately GENERIC (no mention of
# material) so it generalizes past the material-negation case we first tested.
RERANK_INSTRUCT = (
    "Given a product description, decide if the candidate tariff (HS) "
    "classification describes the product."
)
# Reranker on CPU for the same reason as the embedder: coexists with the ~11GB
# vLLM server on a 12GB GPU. ~50 pair-forwards for one query is a couple seconds.
RERANK_DEVICE = "cpu"

# Keep this prompt SHORT. A long, constraint-heavy prompt makes the Thinking
# model reason without terminating (burns the whole token budget, emits no JSON).
# A terse "look, brief reason, output JSON" reliably finishes (finish=stop).
# Think in 3 short sentences max, then output JSON. Do not exceed that — brevity is required.

PROMPT = """You extract product attributes for customs tariff (HS) classification.
RULES:
- Describe ONLY the product. NEVER mention background, surface it rests on,
  lighting, or photo setting.
- State what the object IS (its common article name: e.g. wallet, purse,
  belt, key-case).
- If the item is held on person, then HOW/WHERE a person carries or uses it (pocket, handbag,
  worn, desk).
- Report only visually-verifiable construction and material. Do NOT guess
  fiber content, hide/skin species, or genuine-vs-synthetic — put those in
  uncertain_attributes.
- embedding_description must be ONE tariff-style sentence that leads with the
  article name and its carry/use context (if needed), then outer-surface material, then
  construction. No colours-only, no scene, no filler.
- Use tarrif terminology for categorizing the product.

Output ONLY this JSON (nothing after):
{
  "category": "<common article name>",
  "function": "<how/where carried or used>",
  "visible_construction": "<stitching/closure/fold — only if evident>",
  "visible_materials": ["<outer surface material as it appears>"],
  "embedding_description": "<article name + carry/use context (if needed) + outer-surface
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
    """Embed one QUERY string with the corpus model (L2-normalized, cosine-ready).

    The text is wrapped with the Qwen3-Embedding task instruction (query side);
    the corpus was embedded raw. Runs on CPU (EMBED_DEVICE) to avoid OOM next to
    the vLLM server.
    """
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embed_model.encode(
        [f"{QUERY_INSTRUCT}{text}"], normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


def retrieve(embedding_description: str, k: int = TOP_K, level: str = "subheading"):
    """KNN the embedding_description against vec_embeddings; print the best
    SHOW_N for readability and RETURN all k candidates for reranking. No writes.

    Filtering on `level` is a metadata pre-filter applied INSIDE the vec0 MATCH,
    so the k results are all of that level (default 'subheading' = the HS6 leaf
    rows of the HST corpus, not broad chapter/heading rows). Pre-filtering does
    not change any cosine score — it just curates the candidate pool. Pass
    level=None to search all levels.

    Returns: list of (pct_code, level, distance, description_full), best-first,
    length k. Only the top SHOW_N are printed; the rest go to the reranker.
    """
    con = sqlite3.connect(DB)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)

    qvec = embed(embedding_description)
    blob = struct.pack(f"{DIM}f", *qvec.tolist())
    sql = ("SELECT pct_code, level, distance, description_full "
           "FROM vec_embeddings WHERE embedding MATCH ? AND k = ?")
    params = [blob, k]
    if level:
        sql += " AND level = ?"
        params.append(level)
    sql += " ORDER BY distance"
    rows = con.execute(sql, params).fetchall()
    con.close()

    tag = f" [level={level}]" if level else ""
    print(f"\n{'='*72}\nRETRIEVE: top {SHOW_N} of {len(rows)} cosine candidates"
          f"{tag}\n{'='*72}")
    print(f"query: {embedding_description!r}\n")
    for pct_code, lvl, dist, desc_full in rows[:SHOW_N]:
        # vec0 cosine distance -> similarity = 1 - distance
        print(f"sim {1 - dist:.4f}  {pct_code:11s} [{lvl}]")
        print(f"            {desc_full}\n")
    return rows


_rerank_model = None


def rerank(embedding_description: str, candidates: list, show_n: int = SHOW_N):
    """Re-order cosine `candidates` with Qwen3-Reranker-0.6B and print the top
    `show_n`. `candidates` is what retrieve() returns:
    (pct_code, level, distance, description_full) tuples.

    The reranker is a cross-encoder LLM: it reads (query, document) TOGETHER, so
    unlike the bi-encoder it can attend "not leather" -> "leather" and push the
    wrong material DOWN. The document is the candidate's description_full (the
    same text cosine ranked on). RERANK_INSTRUCT tells it what to judge.

    Returns: list of (score, pct_code, level, description_full), best-first.
    """
    global _rerank_model
    if not candidates:
        print("\n!! no candidates to rerank")
        return []
    if _rerank_model is None:
        import torch
        from sentence_transformers import CrossEncoder
        if RERANK_DEVICE == "cpu":
            # same WSL2 spike guard as the embedder: don't pin every core.
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))
        _rerank_model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)

    docs = [c[3] for c in candidates]  # description_full per candidate
    pairs = [(embedding_description, d) for d in docs]
    scores = _rerank_model.predict(pairs, prompt=RERANK_INSTRUCT)

    ranked = sorted(
        ((float(s), c[0], c[1], c[3]) for s, c in zip(scores, candidates)),
        key=lambda x: -x[0],
    )

    print(f"\n{'='*72}\nRERANK: top {show_n} of {len(candidates)} "
          f"(Qwen3-Reranker-0.6B)\n{'='*72}")
    print(f"query: {embedding_description!r}\n")
    for score, pct_code, lvl, desc_full in ranked[:show_n]:
        print(f"score {score:+.4f}  {pct_code:11s} [{lvl}]")
        print(f"            {desc_full}\n")
    return ranked


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
        max_tokens=14000,
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

    # embed the embedding_description, retrieve a wide cosine pool, then rerank.
    desc = (parsed.get("embedding_description") or "").strip()
    if desc:
        candidates = retrieve(desc)
        rerank(desc, candidates)
    else:
        print("\n!! no embedding_description in output — skipping retrieval")


if __name__ == "__main__":
    main()
