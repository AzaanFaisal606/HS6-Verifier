#!/usr/bin/env python3
"""Phase 2: parse cached Flexport HTS chapter HTML into hst/hst_corpus.db.

HS6 ceiling: emits chapter / heading / subheading rows only. Codes + levels
come from id="hs-<digits>" anchors; names + grouping labels from visible cells.
Indent dots do NOT encode code level (proven unreliable for that), but the dot on
a *bare 6-digit* row DOES reliably mark grouping-label membership: dots=0 =
standalone subheading, dots>=1 = nested under the active label. 10-digit rows
never render dots, so 10-digit-only synthetics carry no signal (_STANDALONE_OVERRIDE).
See docs/superpowers/specs/2026-06-30-hst-corpus-design.md and docs/issues.md.
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
# visual indent marker div: one `·` per nesting level (text-gray-300)
_DOT = "·"

# --- Standalone override (see docs/issues.md Issue 6 "grouping-label leak") ---
# A colSpan grouping label governs only the HS6 rows nested UNDER it; a standalone
# subheading at heading level must NOT inherit it. The group-exit signal is the
# `·` indent dot on the establishing anchor: 0 dots => standalone, >=1 => nested.
# This works for 6-, 8-, AND 10-digit establishers (10-digit rows DO carry the dot,
# just deep in the row markup — see the load-bearing full-chunk count below). So the
# universal rule catches every leak and this override is EMPTY by design. Verified:
# with it empty, description_full is unique across all 7,116 rows. Kept only as a
# hand escape-hatch for any future semantic leak the dot count genuinely can't see.
_STANDALONE_OVERRIDE: set[str] = set()


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

        # A colSpan grouping label (rendered at indent dots=0) governs only the
        # HS6 rows nested UNDER it — those render their bare-6 anchor at dots>=1
        # (e.g. 4202.11 / 4202.12 under "Trunks, suitcases ... school satchels").
        # A *standalone* subheading sits back at heading level: its bare-6 anchor
        # renders at dots=0, and it must NOT inherit the active label (nor may the
        # label leak to following siblings). So a bare-6 row at dots=0 CLEARS the
        # active label. (Deeper national-line labels like "Motors"/"DC" are never
        # picked up here: _LABEL_RE's bare `><div>` only matches dots=0 HS6-group
        # labels; indented labels lead with a `text-gray-300` dot div.) A new
        # label REPLACES the active one. Buffer re-init'd per heading section so
        # labels never leak across headings. 10-digit-only synthetic standalones
        # carry no indent signal and are handled by _STANDALONE_OVERRIDE above.
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
            # Standalone detection (universal, all digit lengths). The `·` indent
            # dot on an establishing row is the nesting depth: a subheading nested
            # under a colSpan grouping label renders >=1 dot, a standalone at
            # heading level renders 0. So dots==0 => this row is NOT under the
            # active label -> clear it. Verified across 6/8/10-digit establishers:
            # nested members (8501.64 under "AC generators"; 0106.11 "Primates"
            # under "Mammals") carry dots>=1; leaks (1902.20 stuffed pasta wrongly
            # under "Uncooked pasta, not stuffed"; 0101.30 "Asses" wrongly under
            # "Horses"; 0102.90 residual "Other" under "Buffalo") carry dots==0.
            # NB: the dot can sit deep in the row markup (mobile-variant cell), so
            # count over the WHOLE chunk, never a truncated prefix. The override
            # remains only for the rare semantic case the dot count can't reach.
            if chunk.count(_DOT) == 0 or six in _STANDALONE_OVERRIDE:
                group_label = None
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
