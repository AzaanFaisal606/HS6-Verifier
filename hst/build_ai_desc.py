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
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DB = str(ROOT / "hst" / "hst_corpus.db")     # codes table (target)
AI_DB = str(ROOT / "hst" / "ai_corpus.db")          # ai_descriptions (few-shot pool)
PROMPT_PATH = str(ROOT / "hst" / "ai_desc_prompt.md")

TEMPERATURE = 0.1          # low: curb invention drift (the "(laptop)" case)
MAX_TOKENS = 200           # output cap; dense rewrites are short
N_PAIRS = 8                # dynamic contrast pairs per call
WRITE_RETRIES = 8

# ---- pluggable inference backends (swap with --backend or REWRITE_BACKEND) ----
# Both speak the OpenAI chat API (Gemini via its OpenAI-compat endpoint), so the
# whole pipeline is backend-agnostic except the thinking-disable knob, which is
# vendor-specific (see thinking_off()). Add a backend = add one dict entry.
BACKENDS = {
    # local vLLM-served 27B NVFP4 on :8001 (the original config)
    "local": {
        "model": "qwen3-vl",
        "base_url": "http://localhost:8001/v1",
        "api_key_env": None,          # local server ignores the key
        "api_key": "x",
        "rpm": 0,                     # 0 = no client-side rate limit (single-seq)
    },
    # Google Gemini via OpenAI-compatible endpoint. Free tier measured on this
    # key: 5 RPM per flash model + a daily RPD cap -> per-row is unusable
    # (thousands of requests). BATCHING (--batch-size 50) collapses 5,756 rows to
    # ~115 requests, well under RPD, ~20 min at 5 RPM. 3.5-flash: newest flash,
    # 1M ctx, same free quota bucket as 2.5-flash (no extra cost).
    "gemini": {
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "api_key": None,              # pulled from api_key_env at runtime
        "rpm": int(os.getenv("GEMINI_RPM", "5")),   # measured free-tier limit
    },
}
MAX_BATCH = 100   # quality ceiling: bigger batches drift / drop items
DEFAULT_BACKEND = os.getenv("REWRITE_BACKEND", "local")


def thinking_off(backend: str) -> dict:
    """Return the extra kwargs that disable model 'thinking' for this backend.
    vLLM/Qwen: chat_template_kwargs.enable_thinking=False.
    Gemini 2.5: thinking_budget=0 nested under google config (extra_body)."""
    if backend == "gemini":
        return {"extra_body": {
            "extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

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


# ---- batch mode: pack many rows into one call, parse JSON array back ----
BATCH_INSTR = (
    "\n\n## BATCH MODE\n"
    "You will receive MULTIPLE numbered items, each an HS6 code and its "
    "true_desc. Rewrite EACH one per ALL the rules and examples above — treat "
    "every item independently, applying the same register.\n"
    "Return ONLY a JSON array, one object per input item IN THE SAME ORDER:\n"
    '[{\"hs6\": \"<the code>\", \"description_ai\": \"<your rewrite>\"}, ...]\n'
    "No prose, no commentary, no markdown code fence — just the raw JSON array. "
    "Include every item; never merge or drop items."
)


def render_batch_user(batch) -> str:
    """batch = list of (code, full). One numbered line per item."""
    return "\n".join(
        f"{i}. HS6 {code} | true_desc: {full}"
        for i, (code, full) in enumerate(batch, 1)
    )


def parse_batch_output(text: str) -> dict:
    """Parse the model's JSON array into {hs6: description_ai}. Tolerates a
    stray ```json fence. Returns {} on unparseable output (caller falls back)."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    # grab the outermost [...] if extra prose leaked in
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b != -1 and b > a:
        s = s[a : b + 1]
    try:
        arr = json.loads(s)
    except Exception:
        return {}
    out = {}
    for obj in arr if isinstance(arr, list) else []:
        if isinstance(obj, dict):
            code = str(obj.get("hs6", "")).strip()
            desc = str(obj.get("description_ai", "")).strip()
            if code and desc:
                out[code] = desc
    return out


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


def resolve_backend(name: str) -> dict:
    if name not in BACKENDS:
        sys.exit(f"unknown backend {name!r}; choose from {list(BACKENDS)}")
    cfg = dict(BACKENDS[name])
    if cfg["api_key_env"]:
        key = os.getenv(cfg["api_key_env"])
        if not key:
            sys.exit(f"backend {name!r} needs env {cfg['api_key_env']} (API key)")
        cfg["api_key"] = key
    cfg["name"] = name
    return cfg


class RateLimiter:
    """Client-side pacing (free-tier RPM). rpm=0 disables."""
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0

    def wait(self):
        if self.interval:
            gap = self.interval - (time.time() - self.last)
            if gap > 0:
                time.sleep(gap)
        self.last = time.time()


def call_model(client, model, sysmsg, usermsg, extra_kwargs, max_tokens,
               limiter, tag=""):
    """One chat call with RPM pacing + 429/5xx backoff-retry. Returns content
    string or None on give-up."""
    for attempt in range(6):
        limiter.wait()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sysmsg},
                    {"role": "user", "content": usermsg},
                ],
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                **extra_kwargs,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in
                            ("429", "RESOURCE_EXHAUSTED", "500", "502", "503",
                             "overloaded", "UNAVAILABLE"))
            if transient and attempt < 5:
                back = min(60, 2 ** attempt * 5)   # 5,10,20,40,60,60 s
                print(f"{tag} transient ({msg[:60]}); backoff {back}s",
                      flush=True)
                time.sleep(back)
                continue
            print(f"{tag} API ERROR: {msg[:120]}", flush=True)
            return None
    return None


def run_per_row(rows, client, model, extra_kwargs, limiter, ai_con,
                system_prompt, total):
    ok = fail = 0
    t0 = time.time()
    for i, (code, full) in enumerate(rows, 1):
        pairs = sample_contrast_pairs(ai_con, code[:2], N_PAIRS)
        sysmsg = system_prompt + render_pairs(pairs)
        ai_desc = call_model(client, model, sysmsg,
                             f"HS6: {code}\ntrue_desc: {full}",
                             extra_kwargs, MAX_TOKENS, limiter,
                             tag=f"[{i}/{total}] {code}")
        if not ai_desc:
            fail += 1
            continue
        if durable_write(code, ai_desc):
            ok += 1
            if i % 25 == 0 or i <= 5:
                rate = i / (time.time() - t0)
                eta = (total - i) / rate / 60 if rate else 0
                print(f"[{i}/{total}] {code} OK | {rate:.2f} rows/s | "
                      f"ETA {eta:.0f} min | {ai_desc[:70]}", flush=True)
        else:
            print(f"[{i}/{total}] {code} WRITE FAILED", flush=True)
            fail += 1
    return ok, fail, (time.time() - t0)


def run_batched(rows, client, model, extra_kwargs, limiter, ai_con,
                system_prompt, total, batch_size):
    ok = fail = 0
    t0 = time.time()
    n_batches = (total + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = rows[b * batch_size:(b + 1) * batch_size]
        # contrast pairs for the batch's dominant chapter (rows are pct_code-
        # ordered, so a batch is chapter-contiguous)
        chapter = batch[len(batch) // 2][0][:2]
        pairs = sample_contrast_pairs(ai_con, chapter, N_PAIRS)
        sysmsg = system_prompt + render_pairs(pairs) + BATCH_INSTR
        # generous out budget: ~40 tokens/row + JSON overhead
        out_cap = min(60000, 120 * len(batch) + 500)
        raw = call_model(client, model, sysmsg, render_batch_user(batch),
                        extra_kwargs, out_cap, limiter,
                        tag=f"[batch {b+1}/{n_batches}]")
        parsed = parse_batch_output(raw) if raw else {}
        b_ok = b_miss = 0
        for code, _full in batch:
            desc = parsed.get(code)
            if desc and durable_write(code, desc):
                ok += 1
                b_ok += 1
            else:
                fail += 1
                b_miss += 1
        rate = (b + 1) * batch_size / (time.time() - t0)
        eta = (total - (b + 1) * batch_size) / rate / 60 if rate else 0
        print(f"[batch {b+1}/{n_batches}] {chapter}xx ok={b_ok} miss={b_miss} "
              f"| ~{rate*60:.0f} rows/min | ETA {eta:.0f} min", flush=True)
    return ok, fail, (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true",
                    help="overwrite all rows (default: only NULL description_ai)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (debug)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND,
                    choices=list(BACKENDS),
                    help=f"inference backend (default {DEFAULT_BACKEND!r}; "
                         f"or set REWRITE_BACKEND)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help=f"rows per model call (1=per-row; capped at {MAX_BATCH} "
                         f"for quality). Batching collapses request count for "
                         f"RPD-limited backends like gemini.")
    args = ap.parse_args()

    batch_size = max(1, min(args.batch_size, MAX_BATCH))
    if args.batch_size > MAX_BATCH:
        print(f"[warn] batch-size clamped to {MAX_BATCH}", flush=True)

    cfg = resolve_backend(args.backend)
    model = cfg["model"]
    limiter = RateLimiter(cfg["rpm"])
    extra_kwargs = thinking_off(cfg["name"])
    system_prompt = load_prompt() + CONTRAST_HEADER

    con = sqlite3.connect(CORPUS_DB, timeout=60)
    add_column_if_missing(con)
    ai_con = sqlite3.connect(AI_DB, timeout=60)
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

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
    mode = f"batched(size={batch_size})" if batch_size > 1 else "per-row"
    print(f"[start] {total} rows to rewrite ({done_already} already done). "
          f"backend={cfg['name']} model={model} mode={mode} temp={TEMPERATURE} "
          f"pairs={N_PAIRS} rpm={cfg['rpm'] or 'unbounded'}", flush=True)

    if not rows:
        print("[done] nothing to do", flush=True)
        return

    if batch_size > 1:
        ok, fail, secs = run_batched(rows, client, model, extra_kwargs, limiter,
                                     ai_con, system_prompt, total, batch_size)
    else:
        ok, fail, secs = run_per_row(rows, client, model, extra_kwargs, limiter,
                                     ai_con, system_prompt, total)
    print(f"[done] ok={ok} fail={fail} in {secs/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
