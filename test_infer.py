
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

DB = Path(__file__).resolve().parent / "hst" / "hst_corpus.db"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DIM = 1024
TOP_K = 50
SHOW_N = 5
EMBED_DEVICE = "cpu"
QUERY_INSTRUCT = (
    "Instruct: Given a product description, retrieve the matching "
    "Harmonized System (HS) tariff classification.\nQuery: "
)

RERANK_INSTRUCT = (
    "Given a product description, decide if the candidate tariff (HS) "
    "classification describes the product."
)
RERANK_DEVICE = "cpu"

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


REINFER_CHAPTERS = 2

REINFER_PROMPT = """You previously described this product for customs tariff (HS)
classification. Below are REAL tariff descriptions retrieved from the HS
schedule. They are shown ONLY as examples of how tariff text is phrased
(register, word order, terminology) — they are NOT necessarily the correct
classification for THIS product and may describe entirely different goods.

SAMPLE TARIFF DESCRIPTIONS (style reference only):
{samples}

Your previous description of the product was:
"{prev_desc}"

TASK: Look at the image again and write ONE new tariff-style sentence describing
the ACTUAL product you see.
RULES:
- Match the STYLE of the samples (formal tariff register, article-name-first,
  terminology) — do NOT copy their wording, article names, materials, or
  classification.
- Describe the product in the image, NOT any of the sample products. If none of
  the samples match the product, ignore their content and describe what you see.
- Lead with the article name + carry/use context, then outer-surface material,
  then construction.
- Do not invent material/species you cannot see.

Output ONLY this JSON (nothing after):
{{
  "embedding_description": "<one tariff-style sentence describing the product>"
}}"""


def reinfer(image_url: str, prev_desc: str, candidates: list,
            n_chapters: int = REINFER_CHAPTERS, verbose: bool = True):
    if not candidates:
        if verbose:
            print("\n!! reinfer: no candidates — keeping previous description")
        return prev_desc

    samples = []
    seen_chapters = set()
    for pct_code, lvl, dist, desc_full in candidates:
        chapter = str(pct_code)[:2]
        if chapter in seen_chapters:
            continue
        seen_chapters.add(chapter)
        samples.append((pct_code, desc_full))
        if len(samples) >= n_chapters:
            break

    samples_block = "\n".join(f"- {desc}" for _, desc in samples)
    prompt = REINFER_PROMPT.format(samples=samples_block, prev_desc=prev_desc)

    if verbose:
        print(f"\n{'='*72}\nREINFER: {len(samples)} distinct-chapter style sample(s)"
              f"\n{'='*72}")
        for pct_code, desc in samples:
            print(f"  [{str(pct_code)[:2]}] {pct_code}: {desc}")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=2500,
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    try:
        new_desc = (json.loads(extract_json(content)).get(
            "embedding_description") or "").strip()
    except json.JSONDecodeError as e:
        if verbose:
            print(f"!! reinfer JSON parse failed: {e} — keeping previous description")
        return prev_desc

    if not new_desc:
        if verbose:
            print("!! reinfer produced no embedding_description — keeping previous")
        return prev_desc

    if verbose:
        print(f"\nprev: {prev_desc!r}")
        print(f"new : {new_desc!r}")
    return new_desc


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
    """Run the flat KNN + reinfer + rerank flow (test_infer's 'normal' pipeline).

    Same logic and params as the CLI. Prints the section blocks only when
    verbose=True; always returns the standard result dict:

        {flow, hs6, description, query, caption, chapters, candidates, meta}

    candidates: [{hs6, description, score, score_type, level}, ...] (top SHOW_N).
    On any failure (bad JSON, no embedding_description, empty pool) returns a
    dict with error set and hs6=None.
    """
    image_url = to_image_url(image)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=2500,
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

    def err(msg_):
        if verbose:
            print(f"\n!! {msg_}")
        return {"flow": "normal", "hs6": None, "description": None,
                "query": None, "caption": None, "chapters": [],
                "candidates": [], "meta": {}, "error": msg_}

    try:
        parsed = json.loads(extract_json(content))
    except json.JSONDecodeError as e:
        return err(f"JSON parse failed: {e}")
    if verbose:
        print(f"\n{'='*72}\nPARSED JSON\n{'='*72}")
        print(json.dumps(parsed, indent=2))

    base_desc = (parsed.get("embedding_description") or "").strip()
    if not base_desc:
        return err("no embedding_description in output")

    # Composed query (small_changes.md 2026-07-07): append function + category
    # to embedding_description so the discriminator (often in `function`, Issue 1)
    # and the article category reach the embedded/reranked text. func+cat come
    # from the ORIGINAL caption JSON (reinfer only rewrites embedding_description),
    # and are appended to BOTH the pre-reinfer and post-reinfer queries.
    func = (parsed.get("function") or "").strip()
    cat = (parsed.get("category") or "").strip()

    def compose(base: str) -> str:
        parts = [base]
        if func:
            parts.append(f"Function: {func}.")
        if cat:
            parts.append(f"Category: {cat}.")
        return " ".join(parts)

    desc = compose(base_desc)
    candidates = retrieve(desc, verbose=verbose)
    new_base = reinfer(image_url, base_desc, candidates, verbose=verbose)
    new_desc = compose(new_base)
    if new_desc != desc:
        candidates = retrieve(new_desc, verbose=verbose)
    ranked = rerank(new_desc, candidates, verbose=verbose)

    if not ranked:
        out = err("empty rerank pool")
        out["query"] = new_desc
        out["caption"] = parsed
        return out

    cands = [{"hs6": norm6(pc), "description": dfull, "score": sc,
              "score_type": "rerank", "level": lvl}
             for sc, pc, lvl, dfull in ranked[:SHOW_N]]
    top = cands[0]
    return {
        "flow": "normal",
        "hs6": top["hs6"],
        "description": top["description"],
        "query": new_desc,
        "caption": parsed,
        "chapters": [],
        "candidates": cands,
        "meta": {"reinfer_prev": base_desc, "reinfer_new": new_base},
    }


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    classify(image, verbose=True)


if __name__ == "__main__":
    main()
