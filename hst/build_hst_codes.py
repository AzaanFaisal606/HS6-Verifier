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
# <h2 class="subheading">DDDD<!-- -->: <!-- -->NAME</h2>
_HEAD_RE = re.compile(
    r'<h2 class="subheading">\s*(\d{4})\s*(?:<!--\s*-->)?:\s*'
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

        labels: list[str] = []
        seen6: set[str] = set()
        for chunk in _TR_SPLIT.split(section):
            lm = _LABEL_RE.search(chunk)
            if lm:
                labels.append(_clean(lm.group(1)))
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
            full_parts = [ch_name, head_name] + labels + ([raw] if raw else [])
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
            labels = []  # flat attach: clear buffer after each emitted HS6

    return rows
