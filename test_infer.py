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
DB = Path(__file__).resolve().parent / "pct" / "pct_corpus.db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
TOP_K = 8

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


def retrieve(embedding_description: str, k: int = TOP_K, level: str = "national"):
    """KNN the embedding_description against vec_embeddings; print top-k with
    full ancestor-chain descriptions. No DB writes.

    Filtering on `level` is a metadata pre-filter applied INSIDE the vec0 MATCH,
    so the k results are all of that level (default 'national' = the real 8-digit
    tariff lines, not broad chapter/heading rows). Pre-filtering does not change
    any cosine score — it just curates the candidate pool. Pass level=None to
    search all levels.
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
    print(f"\n{'='*72}\nTOP {k} CORPUS CANDIDATES (cosine){tag}\n{'='*72}")
    print(f"query: {embedding_description!r}\n")
    for pct_code, level, dist, desc_full in rows:
        # vec0 cosine distance -> similarity = 1 - distance
        print(f"sim {1 - dist:.4f}  {pct_code:11s} [{level}]")
        print(f"            {desc_full}\n")


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
