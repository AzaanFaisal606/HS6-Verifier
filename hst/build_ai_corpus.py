"""Build a corpus of AI-generated product descriptions.

Sends each image to the VLM with the SAME caption prompt as test_infer.py
(minus any chapter selection), parses the JSON, and stores a composed
description (embedding_description + function + category) against the image
name in hst/ai_corpus.db.

Purpose: compare AI-speak descriptions against the pure-tariff HST corpus so
the HST text can later be normalized closer to how models describe products
(combat cosine/reranker underperformance from register mismatch).

Usage:
    python build_ai_corpus.py <image_or_dir>       # dir batch / single file
    python build_ai_corpus.py --map <sample.csv>   # caption a stratified map

  - dir  -> caption every image inside (batch)
  - file -> caption just that one image
  - --map -> read a sample_map.csv (from the ABO stratified sampler) and caption
             each row's `abs_path`. Stores the row's `image_id` as image_name and
             its `product_type` alongside the desc (product_type is metadata for
             the later true-HS6 labeling phase).

The `true_hs6` / `true_desc` columns are left NULL by this script — they are
filled in a LATER labeling phase (a grounded subagent views each image, picks the
correct HS6 from the `codes` table, and back-fills the code + its description_full).

Skips (model_name, image_name) pairs already stored (resumable). In map mode the
skip key is `image_id`; in dir/file mode it is the file basename.
Swap the model by editing MODEL below.
"""

import base64
import csv
import json
import mimetypes
import sqlite3
import sys
from pathlib import Path

from openai import OpenAI

# --- config (swappable) ---
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL = "Qwen/Qwen3.5-4B-BF16"   # full model name + quant = served-model-name (stored as model_name)

MAX_TOKENS = 2500
TEMPERATURE = 0.1
ENABLE_THINKING = False
RETRIES = 3          # dirty JSON is nondeterministic at temp 0.1; retry before failing

DB = Path(__file__).resolve().parent / "ai_corpus.db"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# PROMPT copied verbatim from test_infer.py (plain caption, no chapter select).
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


def compose(parsed: dict) -> str:
    """Composed desc = embedding_description + Function + Category (test_infer)."""
    base = (parsed.get("embedding_description") or "").strip()
    if not base:
        return ""
    parts = [base]
    func = (parsed.get("function") or "").strip()
    cat = (parsed.get("category") or "").strip()
    if func:
        parts.append(f"Function: {func}.")
    if cat:
        parts.append(f"Category: {cat}.")
    return " ".join(parts)


def init_db(con: sqlite3.Connection):
    con.execute(
        """CREATE TABLE IF NOT EXISTS ai_descriptions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name   TEXT NOT NULL,
            image_name   TEXT NOT NULL,
            product_type TEXT,
            desc         TEXT NOT NULL,
            true_hs6     TEXT,
            true_desc    TEXT,
            UNIQUE(model_name, image_name)
        )"""
    )
    con.commit()


def _loads_lenient(raw: str):
    """Parse JSON, with escalating repair for the 4B model's malformed output.

    1. strict json.loads
    2. strip trailing commas before } / ]
    3. SALVAGE: we only need embedding_description / function / category for the
       stored desc. The model reliably breaks `uncertain_attributes` (emits
       `"key": "value"` entries inside a JSON array, which is invalid), which
       blocks the whole parse even though we never use that field. So on total
       failure, regex-pull just the three fields we need and return a minimal
       dict. Raises JSONDecodeError only if embedding_description can't be found.
    """
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",(\s*[}\]])", r"\1", raw)  # drop trailing commas
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # salvage the 3 fields we actually store
    def field(name):
        m = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        return (m.group(1) if m else "").encode().decode("unicode_escape",
                                                          "ignore")
    ed = field("embedding_description")
    if not ed:
        raise json.JSONDecodeError("no embedding_description salvageable", raw, 0)
    return {
        "embedding_description": ed,
        "function": field("function"),
        "category": field("category"),
    }


def caption(image_url: str) -> dict | None:
    """Caption one image. The 4B model emits dirty JSON ~12% of the time at
    temp 0.1 (stray quote/comma mid-string); the fault is NON-deterministic, so
    we retry up to RETRIES times before giving up, with a lenient repair pass."""
    last_err = None
    for attempt in range(1, RETRIES + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            return _loads_lenient(extract_json(content))
        except json.JSONDecodeError as e:
            last_err = e
            if attempt < RETRIES:
                print(f"  .. JSON parse failed (attempt {attempt}/{RETRIES}), retrying")
    print(f"  !! JSON parse failed after {RETRIES} attempts: {last_err}")
    return None


def collect_items(arg: str) -> list[tuple]:
    """Return work items (image_name, abs_path, product_type) for dir/file mode.
    product_type is None here (only the map carries it)."""
    p = Path(arg)
    if p.is_dir():
        files = sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in IMG_EXTS)
        return [(f.name, str(f), None) for f in files]
    if p.is_file():
        return [(p.name, str(p), None)]
    print(f"!! not a file or directory: {arg}")
    return []


def collect_map_items(csv_path: str) -> list[tuple]:
    """Read a sample_map.csv -> work items (image_id, abs_path, product_type).
    image_id is stored as image_name (stable key back to the map)."""
    items = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            abs_path = row.get("abs_path", "").strip()
            image_id = row.get("image_id", "").strip()
            if not abs_path or not image_id:
                continue
            if not Path(abs_path).exists():
                print(f"  !! map path missing, skipping: {abs_path}")
                continue
            items.append((image_id, abs_path, row.get("product_type") or None))
    return items


def main():
    if len(sys.argv) < 2:
        print("usage: python build_ai_corpus.py <image_or_dir>")
        print("       python build_ai_corpus.py --map <sample.csv>")
        sys.exit(1)

    if sys.argv[1] == "--map":
        if len(sys.argv) < 3:
            print("usage: python build_ai_corpus.py --map <sample.csv>")
            sys.exit(1)
        items = collect_map_items(sys.argv[2])
    else:
        items = collect_items(sys.argv[1])

    if not items:
        print("no images to process")
        return
    print(f"work items: {len(items)}")

    con = sqlite3.connect(DB)
    init_db(con)

    stored = skipped = failed = 0
    for i, (name, path, product_type) in enumerate(items, 1):
        exists = con.execute(
            "SELECT 1 FROM ai_descriptions WHERE model_name = ? AND image_name = ?",
            (MODEL, name),
        ).fetchone()
        if exists:
            print(f"[skip]  [{i}/{len(items)}] {MODEL} | {name} (already stored)")
            skipped += 1
            continue

        pt = f" ({product_type})" if product_type else ""
        print(f"[caption] [{i}/{len(items)}] {name}{pt} ...")
        try:
            parsed = caption(to_image_url(path))
        except Exception as e:
            print(f"  !! request failed: {e}")
            failed += 1
            continue

        if parsed is None:
            failed += 1
            continue

        desc = compose(parsed)
        if not desc:
            print(f"  !! no embedding_description — skipping {name}")
            failed += 1
            continue

        con.execute(
            "INSERT INTO ai_descriptions (model_name, image_name, product_type, desc) "
            "VALUES (?, ?, ?, ?)",
            (MODEL, name, product_type, desc),
        )
        con.commit()
        stored += 1
        print(f"[stored] {MODEL} | {name}")
        print(f"         {desc}")

    con.close()
    print(f"\n{'='*60}")
    print(f"done: {stored} captioned, {skipped} skipped, {failed} failed")
    print(f"db: {DB}")


if __name__ == "__main__":
    main()
