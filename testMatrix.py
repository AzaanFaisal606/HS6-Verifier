#!/usr/bin/env python
"""testMatrix.py — standing 32-image ABO regression harness.

Drives the full retrieval flow over every image in the matrix set and writes a
structured log (JSONL + a rendered markdown table block) that you then read and
append to docs/pipeline-matrix-results.md. Replaces the per-run scratchpad.

It REUSES the live client modules' own helpers (embed / retrieve / rerank /
reinfer / extract_json / PROMPT) so the harness scores exactly what the client
would — it only adds structured capture (expected-HS6 rank in each pool, the
gated chapters, the embedded query string) that the CLI clients print but don't
return.

Two knobs you actually pick each run:
  --script {normal,dense,ensemble}   which client flow (default normal)
  --model  NAME                      served model label (default qwen3-vl)

The served-model NAME depends on what serve.sh currently has up; the OpenAI
endpoint served-model-name stays `qwen3-vl` regardless, so --model rarely
changes — --port is what moves (27B NVFP4 serves on :8001, the 4B on :8000).

  ~/miniforge3/envs/vision/bin/python testMatrix.py --script normal --port 8001

Per image it captures, matching the md table columns:
  #, product, expected_hs6, cosine#1, rerank#1, rank_cos, rank_rr,
  chapters (dense only), embedding_description (the embedded/reranked query).

Symbols on each stage's #1 are computed here: exact 6-digit ✅, right 4-digit
heading 🔵, neither ❌; trailing ! when the exact expected HS6 is absent from the
whole candidate pool.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX_DIR = ROOT / "images" / "abo" / "matrix"

# filename stem -> (product label, hand-nominated expected HS6). Ground truth is
# kept stable across runs (see CLAUDE.md); edit only if a code is genuinely wrong.
GROUND_TRUTH = {
    "chvar_safety_B071PDTMR5":       ("nitrile gloves",               "4015.19"),
    "ch33_beauty_B083V9N454":        ("facial mask (skincare)",       "3304.99"),
    "ch42_handbag_B078M7SX69":       ("leather handbag",              "4202.21"),
    "ch42_phonecase_B0853X2F4M":     ("phone case",                   "4202.32"),
    "ch42_suitcase_B07BL6RDW7":      ("soft-side suitcase",           "4202.12"),
    "ch48_office_B081QP9GW3":        ("gummed sticky notes",          "4811.90"),
    "ch49_wallart_B073P5QKL4":       ("framed wall-art print",        "4911.91"),
    "ch57_rug_B07B4YVDLQ":           ("area rug",                     "5702.42"),
    "ch63_bedsheet_B07KZ6PC16":      ("bed sheet (printed)",          "6302.21"),
    "ch63_bedsheet2_B07PHFVPZR":     ("bed sheet set",                "6302.31"),
    "ch64_boot_B06XCPVVPS":          ("chelsea boot",                 "6403.91"),
    "ch64_sandal_B076RFQ53M":        ("block-heel sandal",            "6403.99"),
    "ch64_shoe_B06X9STHNG":          ("leather loafer",               "6403.99"),
    "ch65_hat_B07RF9QMHS":           ("knit beanie",                  "6505.00"),
    "ch69_cup_B0896LJNLH":           ("glass tumbler",                "7013.37"),
    "ch69_cup2_B0896KVCJY":          ("glass tumbler",                "7013.37"),
    "ch71_earring_B013URKEGY":       ("gold hoop earrings",           "7113.19"),
    "ch71_necklace_B075LT7F8X":      ("cord choker necklace",         "7117.90"),
    "ch71_ring_B01N26F2NT":          ("silver CZ ring",               "7113.11"),
    "ch73_kitchenware_B07MRDS4TG":   ("copper jug & bottle",          "7418.10"),
    "ch83_handle_B07J4HXTQH":        ("door lever lockset",           "8301.40"),
    "ch85_lightbulb_B07WJBFJZZ":     ("LED bulb",                     "8539.50"),
    "ch94_chair_B07TMH6289":         ("recliner chair",               "9401.61"),
    "ch94_lamp_B073P3NK7T":          ("table lamp",                   "9405.21"),
    "ch94_ottoman_B075X61WKJ":       ("leather ottoman",              "9401.61"),
    "ch94_sofa_B075X4QMW7":          ("velvet sofa",                  "9401.61"),
    "ch94_table_B072ZLCB3M":         ("wood/metal side table",        "9403.60"),
    "ch95_sportinggoods_B07SS9LXLF": ("weight-lifting grips",         "9506.91"),
    "chfood_grocery_B07QC4DZ2L":     ("light cream (dairy)",          "0401.50"),
    "chvar_outdoor_B01DAOM5SW":      ("gas patio heater",             "7321.81"),
    "chvar_petsupply_B07HJ9WBTY":    ("plush cat cave",               "6307.90"),
    "ch30_healthcare_B07B9VR6XT":    ("back-support brace",           "9021.10"),
}


def norm6(code: str) -> str:
    """Normalize an HS code to its 6-digit dotted form for comparison."""
    d = "".join(ch for ch in str(code) if ch.isdigit())[:6]
    return f"{d[:4]}.{d[4:6]}" if len(d) >= 6 else str(code)


def mark(pred: str, expected: str) -> str:
    """✅ exact HS6, 🔵 right heading wrong subheading, ❌ neither."""
    p, e = norm6(pred), norm6(expected)
    if p == e:
        return "✅"
    if p[:4] == e[:4]:
        return "🔵"
    return "❌"


def rank_of(expected: str, pool) -> int:
    """1-indexed position of expected HS6 in an ordered candidate pool, else 0.
    Pool rows are the client's tuples; pct_code is index 0 (cosine) or 1 (rerank)."""
    e = norm6(expected)
    for i, row in enumerate(pool, 1):
        code = row[0] if not isinstance(row[0], float) else row[1]
        if norm6(code) == e:
            return i
    return 0


def patch_endpoint(mod, model: str, port: int):
    """Repoint an imported client module at the chosen served endpoint."""
    from openai import OpenAI
    mod.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    mod.MODEL = model


def patch_column(mod, column: str, rerank_doc: str | None = None):
    """Point the client's retrieval (and optionally its reranker) at a corpus
    register.

    `column` sets the vector column the cosine KNN searches, plus the doc text
    paired with it. `rerank_doc` overrides ONLY what the cross-encoder judges —
    the two stages prefer different registers (dense rewrite retrieves better,
    the full chain reranks better), so they are decoupled. Only clients that
    expose VEC_DOC_PAIRS (test_infer_dense) support this.
    """
    pairs = getattr(mod, "VEC_DOC_PAIRS", None)
    if pairs is None:
        if column != "embedding" or rerank_doc:
            sys.exit(f"{mod.__name__} has no swappable corpus column "
                     f"(--column/--rerank-doc unsupported)")
        return
    if column not in pairs:
        sys.exit(f"unknown --column {column!r}; pick one of {', '.join(pairs)}")
    mod.VEC_COLUMN = column
    mod.DOC_COLUMN = pairs[column]
    if rerank_doc:
        if rerank_doc not in pairs.values():
            sys.exit(f"unknown --rerank-doc {rerank_doc!r}; pick one of "
                     f"{', '.join(pairs.values())}")
        mod.RERANK_DOC_COLUMN = rerank_doc


def caption(mod, image_url: str, prompt: str):
    """One caption call through the module's own client/MODEL. Returns parsed
    JSON dict or None."""
    resp = mod.client.chat.completions.create(
        model=mod.MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=2500,
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(mod.extract_json(content))
    except json.JSONDecodeError:
        return None


def compose(parsed: dict, base: str) -> str:
    """Append function + category to the base embedding_description, matching the
    live clients' composed-query behaviour."""
    func = (parsed.get("function") or "").strip()
    cat = (parsed.get("category") or "").strip()
    parts = [base]
    if func:
        parts.append(f"Function: {func}.")
    if cat:
        parts.append(f"Category: {cat}.")
    return " ".join(parts)


def run_normal(mod, image_path: Path):
    """test_infer.py flow: caption -> compose -> cosine KNN -> reinfer -> KNN ->
    rerank. Returns capture dict."""
    image_url = mod.to_image_url(str(image_path))
    parsed = None
    for _ in range(2):                      # one retry on empty/unparseable JSON
        parsed = caption(mod, image_url, mod.PROMPT)
        if parsed and (parsed.get("embedding_description") or "").strip():
            break
    if not parsed:
        return {"error": "no JSON from caption"}
    base = (parsed.get("embedding_description") or "").strip()
    if not base:
        return {"error": "no embedding_description"}

    desc = compose(parsed, base)
    candidates = mod.retrieve(desc)
    new_base = mod.reinfer(image_url, base, candidates)
    new_desc = compose(parsed, new_base)
    if new_desc != desc:
        candidates = mod.retrieve(new_desc)
    ranked = mod.rerank(new_desc, candidates)
    return {
        "query": new_desc,
        "cos_pool": candidates,
        "rr_pool": ranked,
        "cos_top": candidates[0][0] if candidates else None,
        "rr_top": ranked[0][1] if ranked else None,
        "chapters": [],
    }


def run_dense(mod, image_path: Path):
    """test_infer_dense.py flow: caption (with chapter picks) -> chapter-gated
    cosine KNN -> rerank. Returns capture dict incl. picked chapters."""
    image_url = mod.to_image_url(str(image_path))
    chapters = mod.load_chapters()
    valid = {c for c, _ in chapters}
    catalog = "\n".join(f"{c} {title}" for c, title in chapters)
    prompt = mod.PROMPT.format(chapter_catalog=catalog, n_chapters=mod.N_CHAPTERS)

    parsed = None
    for _ in range(2):
        parsed = caption(mod, image_url, prompt)
        if parsed and (parsed.get("embedding_description") or "").strip():
            break
    if not parsed:
        return {"error": "no JSON from caption"}
    base = (parsed.get("embedding_description") or "").strip()
    if not base:
        return {"error": "no embedding_description"}
    desc = compose(parsed, base)

    picked, seen = [], set()
    for c in parsed.get("chapters", []):
        c = str(c).strip().zfill(2)
        if c in valid and c not in seen:
            seen.add(c)
            picked.append(c)
    picked = picked[:mod.N_CHAPTERS]
    if not picked:
        return {"error": "no valid chapters", "query": desc}

    candidates = mod.retrieve(desc, picked)
    ranked = mod.rerank(desc, candidates)
    return {
        "query": desc,
        "cos_pool": candidates,
        "rr_pool": ranked,
        "cos_top": candidates[0][0] if candidates else None,
        "rr_top": ranked[0][1] if ranked else None,
        "chapters": picked,
    }


RUNNERS = {"normal": ("test_infer", run_normal),
           "dense": ("test_infer_dense", run_dense)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", choices=list(RUNNERS), default="normal")
    ap.add_argument("--model", default="qwen3-vl",
                    help="served model label (endpoint served-model-name)")
    ap.add_argument("--port", type=int, default=8001,
                    help="local OpenAI endpoint port (27B NVFP4=8001, 4B=8000)")
    ap.add_argument("--column", default="embedding",
                    choices=["embedding", "embedding_own", "embedding_ai"],
                    help="corpus register retrieval+rerank score against: "
                         "embedding (description_full, Runs 7/8), embedding_own, "
                         "or embedding_ai (register-normalized rewrite). "
                         "dense only.")
    ap.add_argument("--rerank-doc",
                    choices=["description_full", "description_own",
                             "description_ai"],
                    help="doc text the CROSS-ENCODER judges, if it should differ "
                         "from --column's paired text. Mixed-field: retrieve on "
                         "embedding_ai, rerank on description_full. dense only.")
    ap.add_argument("--images", default=str(MATRIX_DIR))
    ap.add_argument("--out", default=str(ROOT / "matrix_run.log"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images")
    args = ap.parse_args()

    if args.script == "ensemble":
        print("ensemble not wired in harness yet", file=sys.stderr)
        sys.exit(2)
    mod_name, runner = RUNNERS[args.script]
    mod = importlib.import_module(mod_name)
    patch_endpoint(mod, args.model, args.port)
    patch_column(mod, args.column, args.rerank_doc)

    img_dir = Path(args.images)
    stems = sorted(GROUND_TRUTH)               # stable order for numbering
    if args.limit:
        stems = stems[:args.limit]

    results = []
    for stem in stems:
        matches = list(img_dir.glob(stem + ".*"))
        if not matches:
            print(f"!! missing image for {stem}", file=sys.stderr)
            continue
        product, expected = GROUND_TRUTH[stem]
        print(f"\n########## {stem}  ({product}, expect {expected}) ##########",
              file=sys.stderr)
        t0 = time.time()
        try:
            cap = runner(mod, matches[0])
        except Exception as e:
            cap = {"error": f"{type(e).__name__}: {e}"}
        rec = {"stem": stem, "product": product, "expected": expected,
               "secs": round(time.time() - t0, 1)}
        if "error" in cap and "cos_pool" not in cap:
            rec["error"] = cap["error"]
            rec["query"] = cap.get("query", "")
        else:
            cos_top = cap.get("cos_top")
            rr_top = cap.get("rr_top")
            rec.update({
                "query": cap["query"],
                "chapters": cap["chapters"],
                "cos_top": norm6(cos_top) if cos_top else None,
                "rr_top": norm6(rr_top) if rr_top else None,
                "cos_mark": mark(cos_top, expected) if cos_top else "❌",
                "rr_mark": mark(rr_top, expected) if rr_top else "❌",
                "rank_cos": rank_of(expected, cap["cos_pool"]),
                "rank_rr": rank_of(expected, cap["rr_pool"]),
            })
        results.append(rec)
        print(f"   cos {rec.get('cos_top')} {rec.get('cos_mark','')}  "
              f"rr {rec.get('rr_top')} {rec.get('rr_mark','')}  "
              f"rank {rec.get('rank_cos')}/{rec.get('rank_rr')}  "
              f"{rec['secs']}s", file=sys.stderr)

    write_log(args, results)


def write_log(args, results):
    out = Path(args.out)
    lines = []
    lines.append(f"# testMatrix run — script={args.script} model={args.model} "
                 f"port={args.port} column={args.column} "
                 f"rerank_doc={args.rerank_doc or '(paired)'}")
    lines.append("")
    # JSONL block (machine record)
    lines.append("## JSONL")
    for r in results:
        lines.append(json.dumps(r, ensure_ascii=False))
    lines.append("")
    # summary tallies
    scored = [r for r in results if "cos_top" in r]
    ex_cos = sum(r["cos_mark"] == "✅" for r in scored)
    ex_rr = sum(r["rr_mark"] == "✅" for r in scored)
    hd_cos = sum(r["cos_mark"] in ("✅", "🔵") for r in scored)
    hd_rr = sum(r["rr_mark"] in ("✅", "🔵") for r in scored)
    in_pool = sum(r["rank_cos"] > 0 or r["rank_rr"] > 0 for r in scored)
    n = len(results)
    lines.append(f"## Summary ({n})")
    lines.append(f"exact #1 — cosine {ex_cos}, rerank {ex_rr}")
    lines.append(f"heading-or-better #1 — cosine {hd_cos}, rerank {hd_rr}")
    lines.append(f"exact in pool — {in_pool}/{n}")
    lines.append("")
    # rendered md table (paste-ready for pipeline-matrix-results.md)
    lines.append("## Table")
    lines.append("| # | Product | Expected HS6 | Cosine #1 | Rerank #1 | "
                 "Rank cos / rr | Chapters | Model `embedding_description` |")
    lines.append("|--:|---------|-------------|-----------|-----------|"
                 ":-------------:|----------|------------------------------|")
    for i, r in enumerate(results, 1):
        if "cos_top" not in r:
            lines.append(f"| {i} | {r['product']} | `{r['expected']}` | "
                         f"ERROR: {r.get('error','')} | — | — / — | — | "
                         f"{r.get('query','')} |")
            continue
        pool_bang = "" if (r["rank_cos"] or r["rank_rr"]) else " !"
        chaps = ", ".join(r["chapters"]) if r["chapters"] else "—"
        rc = r["rank_cos"] or "—"
        rr = r["rank_rr"] or "—"
        lines.append(
            f"| {i} | {r['product']}{pool_bang} | `{r['expected']}` | "
            f"`{r['cos_top']}` {r['cos_mark']} | `{r['rr_top']}` {r['rr_mark']} | "
            f"{rc} / {rr} | {chaps} | {r['query']} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}  ({len(results)} rows, exact rr {ex_rr}/{n})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
