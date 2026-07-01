#!/usr/bin/env python3
"""Phase 2: parse cached Flexport HTS chapter HTML into hst/hst_corpus.db.

HS6 ceiling: emits chapter / heading / subheading rows only. Codes + levels
come from id="hs-<digits>" anchors; names + grouping labels from visible cells;
indentation dots are ignored (proven unreliable). See
docs/superpowers/specs/2026-06-30-hst-corpus-design.md.
"""
import argparse
import re
import sqlite3
import sys
from html import unescape
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "hts_cache"
DB = HERE / "hst_corpus.db"
FISCAL_YEAR = "2026"
SEP = " > "

# <h1 ...>Chapter <!-- -->NN<!-- -->: <!-- -->NAME</h1>
_CH_RE = re.compile(
    r"<h1[^>]*>Chapter\s*(?:<!--\s*-->)?\s*(\d{2})\s*(?:<!--\s*-->)?:\s*"
    r"(?:<!--\s*-->)?\s*(.*?)</h1>",
    re.S,
)
# <h2 class="subheading">DDDD[DDDDDD]<!-- -->: <!-- -->NAME</h2>
# Single-line headings render the full 10-digit code in the h2; capture only
# the first 4 digits as the heading code and discard the rest before the colon.
_HEAD_RE = re.compile(
    r'<h2 class="subheading">\s*(\d{4})\d*\s*(?:<!--\s*-->)?:\s*'
    r"(?:<!--\s*-->)?\s*(.*?)</h2>",
    re.S,
)
# split point for table rows
_TR_SPLIT = re.compile(r'<tr data-slot="table-row"')
# grouping label row: empty first cell + colSpan="5" label
_LABEL_RE = re.compile(
    r'colSpan="5"[^>]*><div class="flex items-center"><div>(.*?)</div>', re.S
)
# code row anchor
_ID_RE = re.compile(r'id="hs-(\d+)"')
# own name (desktop variant shown first)
_NAME_RE = re.compile(r'class="hidden md:block">(.*?)<', re.S)


def _clean(s: str) -> str:
    return unescape(re.sub(r"\s+", " ", s)).strip()


def _hs6(digits: str) -> str:
    six = digits[:6]
    return f"{six[:4]}.{six[4:]}"


def parse_chapter(html: str) -> list[dict]:
    rows: list[dict] = []

    m = _CH_RE.search(html)
    if not m:
        raise ValueError("chapter <h1> not found")
    ch_code, ch_name = m.group(1), _clean(m.group(2))
    rows.append(
        dict(
            pct_code=ch_code,
            level="chapter",
            parent_code=None,
            description_raw=ch_name,
            description_full=ch_name,
            is_synthetic=True,
        )
    )

    # bare 6-digit anchors present in the page (drives is_synthetic)
    bare6 = {d for d in _ID_RE.findall(html) if len(d) == 6}

    # process each heading section: from one <h2> to the next (or end)
    heads = list(_HEAD_RE.finditer(html))
    for i, hm in enumerate(heads):
        head_code, head_name = hm.group(1), _clean(hm.group(2))
        if head_code[:2] != ch_code:
            continue  # safety: only this chapter's headings
        rows.append(
            dict(
                pct_code=head_code,
                level="heading",
                parent_code=ch_code,
                description_raw=head_name,
                description_full=SEP.join([ch_name, head_name]),
                is_synthetic=False,
            )
        )
        start = hm.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(html)
        section = html[start:end]

        # One grouping label governs every following HS6 until the next label
        # row appears (no label nesting occurs at the HS6 level — verified across
        # the corpus: zero back-to-back label rows). A new label REPLACES the
        # active group; emitting an HS6 does NOT clear it, so sibling HS6 sharing
        # one colSpan label (e.g. 4202.11 / 4202.12 under "Trunks, suitcases ...
        # school satchels and similar containers") all inherit it. The buffer is
        # re-init'd per heading section, so labels never leak across headings.
        group_label: str | None = None
        seen6: set[str] = set()
        for chunk in _TR_SPLIT.split(section):
            lm = _LABEL_RE.search(chunk)
            if lm:
                group_label = _clean(lm.group(1))
                continue
            idm = _ID_RE.search(chunk)
            if not idm:
                continue
            digits = idm.group(1)
            if len(digits) < 6:
                continue  # heading-level anchor, already handled
            six = _hs6(digits)
            if six in seen6:
                continue  # 8/10-digit child of an already-emitted HS6 -> skip
            seen6.add(six)
            nm = _NAME_RE.search(chunk)
            raw = _clean(nm.group(1)) if nm else ""
            full_parts = (
                [ch_name, head_name]
                + ([group_label] if group_label else [])
                + ([raw] if raw else [])
            )
            rows.append(
                dict(
                    pct_code=six,
                    level="subheading",
                    parent_code=head_code,
                    description_raw=raw,
                    description_full=SEP.join(full_parts),
                    is_synthetic=digits[:6] not in bare6,
                )
            )

    return rows


_SCHEMA = """
CREATE TABLE IF NOT EXISTS codes (
    pct_code         TEXT PRIMARY KEY,
    level            TEXT,
    parent_code      TEXT,
    description_raw  TEXT,
    description_full TEXT,
    cd_percent       TEXT,
    fiscal_year      TEXT,
    is_synthetic     BOOLEAN
)
"""


def build_db(rows: list[dict], db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(_SCHEMA)
        con.executemany(
            "INSERT OR REPLACE INTO codes "
            "(pct_code, level, parent_code, description_raw, description_full, "
            " cd_percent, fiscal_year, is_synthetic) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    r["pct_code"],
                    r["level"],
                    r["parent_code"],
                    r["description_raw"],
                    r["description_full"],
                    None,  # cd_percent deferred
                    FISCAL_YEAR,
                    1 if r["is_synthetic"] else 0,
                )
                for r in rows
            ],
        )
        con.commit()
    finally:
        con.close()


def summarize(ch: str, rows: list[dict]) -> str:
    chap = [r for r in rows if r["level"] == "chapter"]
    heads = [r for r in rows if r["level"] == "heading"]
    subs = [r for r in rows if r["level"] == "subheading"]
    synth = sum(1 for r in subs if r["is_synthetic"])
    name = chap[0]["description_raw"] if chap else "?"
    warns = []
    if not subs:
        warns.append("NO-SUBHEADINGS")
    head_codes = {h["pct_code"] for h in heads}
    parented = {s["parent_code"] for s in subs}
    for hc in head_codes - parented:
        warns.append(f"empty-heading:{hc}")
    for s in subs:
        if not s["description_raw"] and SEP not in s["description_full"].split(
            SEP, 2
        )[-1]:
            warns.append(f"bare-hs6:{s['pct_code']}")
    w = ("  [" + ", ".join(warns) + "]") if warns else ""
    return (
        f"ch {ch}  {name[:40]:<40}  headings={len(heads)} "
        f"subheadings={len(subs)} (synth={synth}){w}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", help="comma list e.g. 01,42,85")
    args = ap.parse_args()

    files = sorted(CACHE.glob("*.html"))
    if args.chapters:
        want = {c.strip().zfill(2) for c in args.chapters.split(",")}
        files = [f for f in files if f.stem in want]
    if not files:
        print("no cached HTML found - run fetch_hts.py first")
        return 1

    all_rows: list[dict] = []
    for f in files:
        try:
            rows = parse_chapter(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"ch {f.stem}: PARSE-ERROR {e}")
            continue
        print(summarize(f.stem, rows))
        all_rows.extend(rows)

    build_db(all_rows, DB)
    print(f"\nwrote {len(all_rows)} rows to {DB.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
