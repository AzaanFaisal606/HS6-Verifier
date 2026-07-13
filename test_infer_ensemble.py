"""Two-VLM ensemble product classification for tariff (HS) codes.

Runs the flat retrieve->rerank pipeline from test_infer.py once per VLM, then
fuses the two reranked lists with Reciprocal Rank Fusion (RRF). Diversity comes
purely from the two different VLMs captioning the same image; the embedder and
reranker are SHARED (one CPU embedder, one CPU reranker), so both models' texts
are scored in the same space.

Flow (2 VLM calls, shared scoring):

    image
      |- caption(VLM_A) -> desc_A -> retrieve -> rerank -> ranked_A
      |- caption(VLM_B) -> desc_B -> retrieve -> rerank -> ranked_B
                                                    |
                                          rrf_fuse([ranked_A, ranked_B])
                                                    |
                                              final top-5

Both VLMs share the plain test_infer.py prompt (no chapter gate, no reinfer).
A model that emits no usable JSON contributes no list; RRF proceeds on whatever
lists survive (degrades to single-model, never crashes).

Usage:
    python test_infer_ensemble.py [image_path_or_url]
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

# --- Ensemble config -------------------------------------------------------
# Two llama.cpp servers (llama-server), one GGUF VLM each, on separate ports.
# Each entry is (base_url, model_id). model_id = the server's --alias.
VLM_MODELS = [
    ("http://localhost:8000/v1", "internvl"),   # InternVL3.5-4B-GGUF
    ("http://localhost:8001/v1", "qwen"),        # Qwen3.5-4B-GGUF
]
RRF_K = 60                               # RRF constant; higher = flatter weighting
FUSION_METHOD = "rrf"                    # only "rrf" implemented

# One OpenAI client per endpoint (base_url differs per model), keyed by base_url.
_clients = {}


def client_for(base_url: str) -> OpenAI:
    if base_url not in _clients:
        _clients[base_url] = OpenAI(base_url=base_url, api_key="EMPTY")
    return _clients[base_url]

DEFAULT_IMG = "test.jpeg"

# --- Corpus retrieval (must match build_embeddings.py) ---------------------
DB = Path(__file__).resolve().parent / "hst" / "hst_corpus.db"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DIM = 1024
TOP_K = 30
SHOW_N = 5
EMBED_DEVICE = "cpu"
RERANK_DEVICE = "cpu"
QUERY_INSTRUCT = (
    "Instruct: Given a product description, retrieve the matching "
    "Harmonized System (HS) tariff classification.\nQuery: "
)
RERANK_INSTRUCT = (
    "Given a product description, decide if the candidate tariff (HS) "
    "classification describes the product."
)

PROMPT = """You extract product attributes for customs tariff (HS) classification for US imports. We are using the HST schedule for codes and descriptions.
RULES:
- Describe ONLY the product. NEVER mention background, surface it rests on,
  lighting, or photo setting.
- State what the object IS (its common article name: e.g. wallet, purse,
  belt, key-case).
- HOW it is used / HOW it functions.
- If the item is held on person, then WHERE a person carries or uses it (pocket, handbag,
  worn, desk).
- Report only visually-verifiable construction and material. Do NOT guess
  fiber content, hide/skin species, or genuine-vs-synthetic — put those in
  uncertain_attributes.
- embedding_description must be ONE tariff-style sentence that leads with the
  article name and its carry/use context, then outer-surface material, then
  construction. No colours-only, no scene, no filler.
- Use tarrif terminology for categorizing the product.

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


def caption(image_url: str, base_url: str, model_id: str, verbose: bool = True):
    """One VLM call for one model on its own endpoint. Returns the parsed JSON
    dict, or None if the model produced no usable JSON (drop this model's vote).
    """
    resp = client_for(base_url).chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=4000,
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    content = (msg.content or "").strip()

    if verbose:
        print(f"{'='*72}\nMODEL OUTPUT [{model_id}]\n{'='*72}")
        if reasoning:
            print("--- thinking ---")
            print(reasoning.strip())
            print("--- output ---")
        print(content)
        print(f"\n[usage] {resp.usage}  finish={resp.choices[0].finish_reason}")

    try:
        parsed = json.loads(extract_json(content))
    except json.JSONDecodeError as e:
        if verbose:
            print(f"!! [{model_id}] JSON parse failed: {e} — dropping this model")
        return None
    if verbose:
        print(f"\n{'='*72}\nPARSED JSON [{model_id}]\n{'='*72}")
        print(json.dumps(parsed, indent=2))
    return parsed


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


def retrieve(embedding_description: str, k: int = TOP_K, level: str = "subheading",
             verbose: bool = True):
    """Flat KNN on `embedding` (description_full), pre-filtered to `level`."""
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

    if verbose:
        tag = f" [level={level}]" if level else ""
        print(f"\n{'='*72}\nRETRIEVE: top {SHOW_N} of {len(rows)} cosine candidates"
              f"{tag}\n{'='*72}")
        print(f"query: {embedding_description!r}\n")
        for pct_code, lvl, dist, desc_full in rows[:SHOW_N]:
            print(f"sim {1 - dist:.4f}  {pct_code:11s} [{lvl}]")
            print(f"            {desc_full}\n")
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


def rrf_fuse(ranked_lists: list, k: int = RRF_K, show_n: int = SHOW_N,
             verbose: bool = True):
    """Reciprocal Rank Fusion over per-model rerank lists.

    Each list is rerank() output: [(score, pct_code, level, desc_full), ...],
    already sorted best-first. RRF score for a code = sum over lists of
    1/(k + rank), rank being its 1-indexed position in that list. A code absent
    from a list contributes nothing from it (no penalty). Fusion key is pct_code;
    level/desc_full are carried from the first list that saw the code. Per-model
    ranks are tracked for display. Returns the fused list sorted best-first:
    [(rrf_score, pct_code, level, desc_full, per_model_ranks), ...].
    """
    lists = [rl for rl in ranked_lists if rl]
    if not lists:
        if verbose:
            print("\n!! no ranked lists to fuse")
        return []

    scores = {}    # pct_code -> rrf score
    meta = {}      # pct_code -> (level, desc_full)
    ranks = {}     # pct_code -> [rank_in_list0, rank_in_list1, ...] (None if absent)

    for li, rl in enumerate(lists):
        for rank, (_score, pct_code, lvl, desc_full) in enumerate(rl, start=1):
            scores[pct_code] = scores.get(pct_code, 0.0) + 1.0 / (k + rank)
            if pct_code not in meta:
                meta[pct_code] = (lvl, desc_full)
            if pct_code not in ranks:
                ranks[pct_code] = [None] * len(lists)
            ranks[pct_code][li] = rank

    fused = sorted(
        ((s, pc, meta[pc][0], meta[pc][1], ranks[pc]) for pc, s in scores.items()),
        key=lambda x: -x[0],
    )

    if verbose:
        print(f"\n{'='*72}\nRRF FUSION: top {show_n} of {len(fused)} fused codes"
              f"  (k={k}, {len(lists)} model list(s))\n{'='*72}")
        for rrf_score, pct_code, lvl, desc_full, per_ranks in fused[:show_n]:
            rank_str = ", ".join(
                f"m{i}=#{r}" if r is not None else f"m{i}=—"
                for i, r in enumerate(per_ranks)
            )
            print(f"rrf {rrf_score:.5f}  {pct_code:11s} [{lvl}]  ({rank_str})")
            print(f"            {desc_full}\n")
    return fused


def norm6(code) -> str:
    """Normalize an HS code to NNNN.NN (6-digit subheading form)."""
    digits = "".join(ch for ch in str(code) if ch.isdigit())[:6]
    return f"{digits[:4]}.{digits[4:6]}" if len(digits) >= 6 else str(code)


def classify(image: str, verbose: bool = False) -> dict:
    """Run the two-VLM RRF ensemble flow.

    Each VLM captions the image, runs retrieve->rerank; the reranked lists are
    fused with RRF. Prints section blocks only when verbose=True; always returns
    the standard result dict:

        {flow, hs6, description, query, caption, chapters, candidates, meta}

    `query` is the first surviving model's embedded description; `caption` is
    that model's parsed JSON. candidates carry the RRF score
    (score_type='rrf') plus per-model contributing ranks in meta. On failure
    (no model produced a usable list) returns a dict with error set, hs6=None.
    """
    def err(msg_):
        if verbose:
            print(f"\n!! {msg_}")
        return {"flow": "ensemble", "hs6": None, "description": None,
                "query": None, "caption": None, "chapters": [],
                "candidates": [], "meta": {}, "error": msg_}

    if FUSION_METHOD != "rrf":
        return err(f"FUSION_METHOD={FUSION_METHOD!r} not implemented (only 'rrf')")

    image_url = to_image_url(image)

    ranked_lists = []
    first_query = None
    first_caption = None
    model_ids = []
    for base_url, model_id in VLM_MODELS:
        parsed = caption(image_url, base_url, model_id, verbose=verbose)
        if parsed is None:
            continue
        desc = (parsed.get("embedding_description") or "").strip()
        if not desc:
            if verbose:
                print(f"\n!! [{model_id}] no embedding_description — dropping this model")
            continue
        if first_query is None:
            first_query = desc
            first_caption = parsed
        candidates = retrieve(desc, verbose=verbose)
        ranked_lists.append(rerank(desc, candidates, verbose=verbose))
        model_ids.append(model_id)

    if not ranked_lists:
        return err("no model produced a usable ranked list — nothing to fuse")

    fused = rrf_fuse(ranked_lists, verbose=verbose)
    if not fused:
        out = err("empty fusion result")
        out["query"] = first_query
        out["caption"] = first_caption
        return out

    cands = [{"hs6": norm6(pc), "description": dfull, "score": rrf_score,
              "score_type": "rrf", "level": lvl, "per_model_ranks": per_ranks}
             for rrf_score, pc, lvl, dfull, per_ranks in fused[:SHOW_N]]
    top = cands[0]
    return {
        "flow": "ensemble",
        "hs6": top["hs6"],
        "description": top["description"],
        "query": first_query,
        "caption": first_caption,
        "chapters": [],
        "candidates": cands,
        "meta": {"models": model_ids, "n_lists_fused": len(ranked_lists)},
    }


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    classify(image, verbose=True)


if __name__ == "__main__":
    main()
