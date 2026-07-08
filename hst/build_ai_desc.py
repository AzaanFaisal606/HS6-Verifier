#!/usr/bin/env python
"""build_ai_desc.py — normalize every HS6 subheading's `description_full` into a
dense, retrieval-register `description_ai`, using the served 27B model.

Per row:
  - fetch the row's `description_full` (the legal chain)
  - sample 8 FRESH dynamic contrast pairs from ai_corpus.db (VLM caption vs
    legalese) — chapter-matched when possible, random backfill otherwise —
    injected as NEGATIVE/CONTRAST reference (do NOT imitate the chatty VLM shape)
  - call the model with the static rewrite prompt + the contrast block
  - durably write `description_ai` back to codes.description_ai (read-back
    confirmed), resumable: rows already filled are skipped.

Run it and leave it — 5,756 rows, one model call each, single-seq 27B (~hours).
Launch in tmux so it survives disconnects:

  tmux new-session -d -s ai_desc \\
    '~/miniforge3/envs/vision/bin/python hst/build_ai_desc.py 2>&1 | tee ai_desc.log'

Idempotent: re-running continues where it stopped (only NULL description_ai rows
are processed). Use --redo to overwrite all.
"""
import argparse
import random
import sqlite3
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DB = str(ROOT / "hst" / "hst_corpus.db")     # codes table (target)
AI_DB = str(ROOT / "hst" / "ai_corpus.db")          # ai_descriptions (few-shot pool)
PROMPT_PATH = str(ROOT / "hst" / "ai_desc_prompt.md")

MODEL = "qwen3-vl"
BASE = "http://localhost:8000/v1"
TEMPERATURE = 0.1          # low: curb invention drift (the "(laptop)" case)
MAX_TOKENS = 200           # output cap; dense rewrites are short
N_PAIRS = 8                # dynamic contrast pairs per call
WRITE_RETRIES = 8

# ---- the contrast-block framing appended after the static prompt ----
CONTRAST_HEADER = """\

## REGISTER-GAP REFERENCE (illustration only — NOT output examples)

Below are real pairs showing the TWO registers your rewrite sits between. Each
pair has a VLM CAPTION (how a vision model chattily describes a product) and the
LEGAL DESC (how the tariff corpus writes the same kind of item). These are here
so you understand the gap you are closing — they are NOT templates to copy.

- The VLM CAPTION is TOO VERBOSE: it leads with "A"/"An", pads with "featuring"
  / "designed for", appends "Function:/Category:" tags, and sometimes guesses
  facts (color, sub-type) the legal text never states. DO NOT imitate its shape
  and DO NOT copy invented facts from it.
- The LEGAL DESC is TOO DENSE and buried in chapter/heading boilerplate.
- YOUR TARGET register sits between them, on the dense side: the plain product
  vocabulary of the caption, but compressed to a canonical noun phrase with
  every discriminator kept and zero invented facts (exactly the WORKED EXAMPLES
  above).

Use these only to calibrate register and borrow plain product nouns. Produce
your answer per the WORKED EXAMPLES and OUTPUT FORMAT, never in the VLM style.

"""


def load_prompt() -> str:
    return Path(PROMPT_PATH).read_text()


def add_column_if_missing(con: sqlite3.Connection) -> None:
    cols = [r[1] for r in con.execute("PRAGMA table_info(codes)")]
    if "description_ai" not in cols:
        con.execute("ALTER TABLE codes ADD COLUMN description_ai TEXT")
        con.commit()
        print("[schema] added codes.description_ai column", flush=True)


def sample_contrast_pairs(ai_con: sqlite3.Connection, chapter: str, k: int):
    """Fresh sample every call. Prefer same-chapter pairs; random backfill.
    Pool already pruned to usable rows (non-empty desc + true_hs6/true_desc)."""
    same = ai_con.execute(
        "SELECT desc, true_desc FROM ai_descriptions "
        "WHERE substr(true_hs6,1,2)=? ",
        (chapter,),
    ).fetchall()
    random.shuffle(same)
    picked = same[:k]
    if len(picked) < k:
        need = k - len(picked)
        seen = {(d, t) for d, t in picked}
        rest = ai_con.execute(
            "SELECT desc, true_desc FROM ai_descriptions WHERE substr(true_hs6,1,2)!=?",
            (chapter,),
        ).fetchall()
        random.shuffle(rest)
        for d, t in rest:
            if (d, t) in seen:
                continue
            picked.append((d, t))
            if len(picked) - (k - need) >= need:
                break
    return picked[:k]


def render_pairs(pairs) -> str:
    out = []
    for i, (vlm, legal) in enumerate(pairs, 1):
        out.append(
            f"Pair {i}:\n"
            f"  VLM CAPTION (too verbose — do NOT imitate): {vlm.strip()}\n"
            f"  LEGAL DESC (too dense): {legal.strip()}"
        )
    return "\n\n".join(out)


def durable_write(code: str, ai_desc: str) -> bool:
    for attempt in range(WRITE_RETRIES):
        try:
            c = sqlite3.connect(CORPUS_DB, timeout=60)
            c.execute("PRAGMA busy_timeout=60000")
            n = c.execute(
                "UPDATE codes SET description_ai=? WHERE pct_code=?",
                (ai_desc, code),
            ).rowcount
            c.commit()
            chk = c.execute(
                "SELECT description_ai FROM codes WHERE pct_code=?", (code,)
            ).fetchone()
            c.close()
            if n == 1 and chk and chk[0] == ai_desc:
                return True
        except Exception as e:
            print(f"  [write err {attempt}] {e}", flush=True)
        time.sleep(1.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true",
                    help="overwrite all rows (default: only NULL description_ai)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (debug)")
    args = ap.parse_args()

    system_prompt = load_prompt() + CONTRAST_HEADER

    con = sqlite3.connect(CORPUS_DB, timeout=60)
    add_column_if_missing(con)
    ai_con = sqlite3.connect(AI_DB, timeout=60)
    client = OpenAI(base_url=BASE, api_key="x")

    where = "level='subheading'"
    if not args.redo:
        where += " AND (description_ai IS NULL OR TRIM(description_ai)='')"
    rows = con.execute(
        f"SELECT pct_code, description_full FROM codes WHERE {where} ORDER BY pct_code"
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    done_already = con.execute(
        "SELECT COUNT(*) FROM codes WHERE level='subheading' "
        "AND description_ai IS NOT NULL AND TRIM(description_ai)!=''"
    ).fetchone()[0]
    print(f"[start] {total} rows to rewrite ({done_already} already done). "
          f"model={MODEL} temp={TEMPERATURE} pairs={N_PAIRS}", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, (code, full) in enumerate(rows, 1):
        chapter = code[:2]
        pairs = sample_contrast_pairs(ai_con, chapter, N_PAIRS)
        sysmsg = system_prompt + render_pairs(pairs)
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sysmsg},
                    {"role": "user", "content": f"HS6: {code}\ntrue_desc: {full}"},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            ai_desc = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[{i}/{total}] {code} API ERROR: {e}", flush=True)
            fail += 1
            continue

        if not ai_desc:
            print(f"[{i}/{total}] {code} EMPTY output — skipped", flush=True)
            fail += 1
            continue

        if durable_write(code, ai_desc):
            ok += 1
            if i % 25 == 0 or i <= 5:
                rate = i / (time.time() - t0)
                eta_min = (total - i) / rate / 60 if rate else 0
                print(f"[{i}/{total}] {code} OK | {rate:.2f} rows/s | "
                      f"ETA {eta_min:.0f} min | {ai_desc[:70]}", flush=True)
        else:
            print(f"[{i}/{total}] {code} WRITE FAILED", flush=True)
            fail += 1

    dt = (time.time() - t0) / 60
    print(f"[done] ok={ok} fail={fail} in {dt:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
