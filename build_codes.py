#!/usr/bin/env python3
"""Parse the FBR Pakistan Customs Tariff PDF into a normalized SQLite `codes` table.

Pure data engineering — no models. See pct-corpus-handoff.md for the spec and
docs/superpowers/specs / plan for scope decisions. Builds ONE table: `codes`.

Pipeline:
  1. pdftotext -layout  -> flowing 3-column text (PCT CODE | DESCRIPTION | CD %)
  2. line walk: classify each line as heading / leaf / dash-label / continuation,
     maintaining a dash-stack of grouping labels (keyed by dash-depth).
  3. normalize: emit national + heading rows, synthesize subheading (6-digit) and
     chapter (2-digit) rows, wire parent_code chain, build description_full.

Hierarchy is derived from the CODE digits, never the dash count (handoff section 3).
"""

import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE / "PTC-2025-26.pdf"
DB = HERE / "pct_corpus.db"
CHAPTER_TITLES_MD = HERE / "docs" / "chapter-titles.md"
FISCAL_YEAR = "2025-26"

# A leaf row:  XXXX.XXXX <description...> [<CD>]
LEAF_RE = re.compile(r"^\s*(\d{4})\.(\d{4})\s*(.*?)\s*$")
# A heading row:  XX.XX <description...> [<CD>]
HEADING_RE = re.compile(r"^\s*(\d{2})\.(\d{2})\s*(.*?)\s*$")
# Trailing CD column: 2+ spaces then an integer at end of line.
CD_RE = re.compile(r"\s{2,}(\d+)\s*$")
# A dash bullet line carrying NO code: leading dashes (with spaces between) + text.
DASH_RE = re.compile(r"^\s*(-(?:\s*-)*)\s*(.*?)\s*$")

NOISE = (
    "PAKISTAN CUSTOMS TARIFF",
    "PCT CODE",
)

# WCO heading titles for headings that are NOT printed as their own XX.XX row in
# the tariff body (they appear only via their leaf lines, so the build has to
# synthesize the heading row). Backfilled here so the row is not title-less.
# See docs/issues.md (issue 2).
HEADING_TITLE_OVERRIDES = {
    "11.03": "Cereal groats, meal and pellets.",
}


def extract_text(pdf_path: Path) -> str:
    out = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout


def load_chapter_titles(md_path: Path) -> dict[str, str]:
    """Parse the markdown table -> {chapter_2digit: title}."""
    titles: dict[str, str] = {}
    for line in md_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        ch, title = cells[0], cells[1]
        if re.fullmatch(r"\d{2}", ch):
            titles[ch] = title
    return titles


def split_cd(text: str) -> tuple[str, str | None]:
    """Split a description+CD string into (description, cd or None)."""
    m = CD_RE.search(text)
    if m:
        return text[: m.start()].rstrip(), m.group(1)
    return text.rstrip(), None


# Specific-duty tails that pdftotext sometimes smears into the description for the
# ~19 worst-mangled Rupee-rate rows (ch 15 oils, a few electronics). Best-effort:
# pull the tail out of the description into cd_percent. The source text itself is
# garbled for these (interleaved fragments), so the captured rate is approximate.
SPECIFIC_DUTY_TAIL = re.compile(
    r"\s+((?:Rs\.?\s*[\d /]*(?:Rs\.?\s*[\d /]*)*"
    r"(?:/?\s*M\s*T|/MT|per\s+meter|per\s+set|/set|/K\w*)?)+\s*[\d /MT]*)\s*$",
    re.IGNORECASE,
)


def extract_specific_duty(desc: str, existing_cd):
    """If desc carries a trailing Rs./MT specific-duty smear, strip it out and
    return (clean_desc, cd). Leaves clean descriptions untouched."""
    if "Rs." not in desc and "Rs " not in desc:
        return desc, existing_cd
    m = SPECIFIC_DUTY_TAIL.search(desc)
    if not m:
        return desc, existing_cd
    tail = re.sub(r"\s+", " ", m.group(1)).strip()
    clean = desc[: m.start()].rstrip()
    return clean, (existing_cd or tail)


def strip_dashes(text: str) -> str:
    """Remove a leading dash-bullet run ('- - -') and collapse inner whitespace."""
    m = DASH_RE.match(text)
    label = m.group(2) if (m and m.group(1)) else text
    return re.sub(r"\s+", " ", label).strip()


def dash_depth(text: str) -> int | None:
    """Count of leading dashes if the line is a dash bullet, else None."""
    m = re.match(r"^\s*(-(?:\s*-)*)", text)
    if not m:
        return None
    return m.group(1).count("-")


class Leaf:
    __slots__ = ("code", "desc_parts", "cd", "depth", "stack", "label_before")

    def __init__(self, code: str, desc: str, cd: str | None):
        self.code = code            # "8543.7010"
        self.desc_parts = [desc]    # accumulates wrapped continuation lines
        self.cd = cd
        self.depth = 0              # own dash count
        self.stack = {}             # dash-stack snapshot above this leaf
        self.label_before = None    # dash-label pushed immediately before

    @property
    def desc(self) -> str:
        return re.sub(r"\s+", " ", " ".join(p for p in self.desc_parts if p)).strip()


class Heading:
    __slots__ = ("code", "desc_parts", "cd")

    def __init__(self, code: str, desc: str, cd: str | None):
        self.code = code            # "85.43"
        self.desc_parts = [desc]
        self.cd = cd

    @property
    def desc(self) -> str:
        return re.sub(r"\s+", " ", " ".join(p for p in self.desc_parts if p)).strip()


class _DashLabel:
    __slots__ = ("depth", "desc_parts")

    def __init__(self, text: str, depth: int):
        self.depth = depth
        self.desc_parts = [text]

    @property
    def desc(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.desc_parts)).strip()


def parse_with_labels(text: str):
    """Walk the layout text and attach, to each leaf, the dash-label context above
    it. Returns the ordered stream of (kind, obj) — kind in {heading,leaf,label}.

    A running dash-stack (depth -> label) reconstructs the un-coded grouping lines
    that sit between a heading and its leaves (handoff section 3). On each new
    bullet we replace that depth's entry and clear anything deeper. Each leaf and
    each label is annotated with:
      - .depth        : its own dash count (None/0 for headings)
      - .stack        : snapshot of the dash-stack ABOVE it (depth -> label)
      - .label_before : the dash-label pushed immediately before it (or None)
    """
    stream = []
    last = None
    dash_stack: dict[int, str] = {}
    label_before: str | None = None  # nearest dash-label since the last code row

    def snapshot():
        return dict(dash_stack)

    for raw in text.splitlines():
        if not raw.strip():
            last = None
            continue
        if any(n in raw for n in NOISE):
            continue
        if raw.strip() in ("2025-26", "CD (%)", "DESCRIPTION"):
            continue
        # Deleted/reserved heading placeholder, e.g. "[05.03]" — not a row.
        if re.fullmatch(r"\[\d{2}\.\d{2}\]", raw.strip()):
            continue

        # Real PCT-code rows sit in the left-hand PCT CODE column (indent <= 2)
        # AND carry description text on the same line. A bare "8703.2113," or
        # "87.03:" at indent 0 is a code REFERENCE de-indented inside a wrapped
        # description block (rampant in ch 87 component lists) — NOT a row.
        # Such lines fall through to the continuation branch below; this prevents
        # them hijacking a real heading/leaf (e.g. "87.03:" overwriting the
        # "Motor cars..." title with component-list text).
        if re.match(r"^\s{0,2}\d{4}\.\d{4}\s+\S", raw):
            m = LEAF_RE.match(raw)
            desc, cd = split_cd(m.group(3))
            leaf = Leaf(f"{m.group(1)}.{m.group(2)}", strip_dashes(desc), cd)
            leaf.depth = dash_depth(desc) or 0
            leaf.stack = snapshot()
            leaf.label_before = label_before
            stream.append(("leaf", leaf))
            last = leaf
            label_before = None
            continue

        if re.match(r"^\s{0,2}\d{2}\.\d{2}\s+\S", raw):
            m = HEADING_RE.match(raw)
            desc, cd = split_cd(m.group(3))
            h = Heading(f"{m.group(1)}.{m.group(2)}", desc.strip(), cd)
            stream.append(("heading", h))
            last = h
            dash_stack = {}          # new heading resets grouping context
            label_before = None
            continue

        depth = dash_depth(raw)
        if depth is not None:
            text_lbl = strip_dashes(raw)
            lbl = _DashLabel(text_lbl, depth)
            # replace this depth, clear deeper levels (new bullet at this depth)
            for d in [d for d in dash_stack if d >= depth]:
                del dash_stack[d]
            dash_stack[depth] = text_lbl
            label_before = text_lbl
            stream.append(("label", lbl))
            last = lbl
            continue

        # Specific-duty CD fragment: some rows (mostly ch 15 oils, ch 24 tobacco)
        # carry a Rupee-per-tonne rate instead of a percent. pdftotext drops it on
        # a separate FAR-RIGHT line ("Rs.10550", "/ MT", split digits) below the
        # code. Capture those into the row's cd_percent, NOT its description.
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if (
            indent >= 45
            and last is not None
            and isinstance(last, (Leaf, Heading))
            and re.fullmatch(
                r"(Rs\.?\s*\d*|/?\s*MT|/MT|\d{2,5}|\d+\s*/\s*K\w*|\d+\s*/|g|/?set)",
                stripped, re.IGNORECASE,
            )
        ):
            frag = re.sub(r"\s+", "", stripped)
            last.cd = (last.cd or "") + frag if last.cd else frag
            continue

        if last is not None:
            last.desc_parts.append(re.sub(r"\s+", " ", raw.strip()))
            # keep stacked label text in sync if a label wrapped
            if isinstance(last, _DashLabel) and last.depth in dash_stack:
                dash_stack[last.depth] = last.desc
                label_before = last.desc

    return stream


def _ordered_labels(stack: dict[int, str]) -> list[str]:
    """Stack {depth: label} -> labels ordered shallow->deep. Depth VALUE is not
    trusted (gaps/non-contiguous are normal, see 8544); only the ORDER is."""
    return [stack[d] for d in sorted(stack)]


def subheading_labels(stream) -> dict[str, dict]:
    """Per 6-digit subheading (XXXX.YY): its own description_raw plus the
    above-HS6 label path (shared group labels). See design doc.

    Rule:
      - group leaves sharing XXXX.YY. Their ordered label stacks share a common
        PREFIX = the labels above the digit-5/6 split (necessarily shared by all
        leaves in the subheading). That prefix is the subheading's label path.
      - the subheading's own description_raw is:
          * the leaf's own label when the group is a single leaf whose own label
            sits at the prefix boundary (4402: "Of bamboo"; 0101.30: "Asses"), or
          * the first label DEEPER than the shared prefix when the leaves split
            below HS6 (4202.11: depth-2 "With outer surface of leather..."), or
          * "" when there is nothing to name it beyond the prefix.
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for kind, obj in stream:
        if kind == "leaf" and not obj.code.endswith(".0000"):
            groups[obj.code[:7]].append(obj)  # "XXXX.YY"

    out: dict[str, dict] = {}
    for sub, leaves in groups.items():
        # full ordered ancestry per leaf = stack labels (above the leaf's own depth)
        # + the leaf's own inline label.
        # When a leaf's own dash depth > 0, any stack entry at depth >= leaf.depth
        # is stale (it belongs to a sibling or was overwritten by the leaf itself);
        # exclude those to get the true ancestry above this leaf.
        ancestries = [
            _ordered_labels(
                {d: v for d, v in lf.stack.items() if lf.depth == 0 or d < lf.depth}
            ) + [extract_specific_duty(lf.desc, None)[0]]
            for lf in leaves
        ]
        # shared prefix across the group (label equality)
        prefix: list[str] = []
        for cols in zip(*ancestries):
            if all(c == cols[0] for c in cols):
                prefix.append(cols[0])
            else:
                break
        # the subheading's own name = the first DIVERGING label if any, else the
        # last shared label split off the path, else "".
        first = ancestries[0]
        if len(prefix) < len(first):
            # Only use the first diverging element as subheading raw if ALL other
            # leaves have the SAME element at that position (meaning it's a genuine
            # sub-group label that identifies the subheading). If leaves diverge in
            # different directions (e.g. 8544.30: 3011/3019 under "Of a kind..." but
            # 3090 directly under "Other"), there is no single subheading name — use "".
            candidate = first[len(prefix)]
            if all(
                len(a) > len(prefix) and a[len(prefix)] == candidate
                for a in ancestries[1:]
            ):
                raw = candidate               # all agree: this label names the subheading
                path = prefix
            else:
                raw = ""                      # leaves diverge; no single sub name
                path = prefix
        elif prefix:
            raw = prefix[-1]                  # single-leaf group: own label is last shared
            path = prefix[:-1]
        else:
            raw = ""
            path = []
        out[sub] = {"raw": raw, "path": path}
    return out


def national_label_path(leaf, subheading_chain: list[str]) -> list[str]:
    """Group labels strictly between the subheading and this leaf's own line.

    subheading_chain = subheading path + [subheading raw]. We drop that prefix
    from the leaf's full ordered ancestry and also drop the leaf's OWN inline
    label (the last element), leaving the below-HS6 national grouping labels.
    """
    own = extract_specific_duty(leaf.desc, None)[0]
    filtered_stack = {d: v for d, v in leaf.stack.items() if leaf.depth == 0 or d < leaf.depth}
    full = _ordered_labels(filtered_stack) + [own]
    # strip the subheading chain prefix where it matches
    i = 0
    for sc in subheading_chain:
        if i < len(full) and full[i] == sc:
            i += 1
        else:
            break
    tail = full[i:]
    # drop the leaf's own inline label (last element) if present
    if tail and tail[-1] == own:
        tail = tail[:-1]
    return tail


def build_db(stream, chapter_titles: dict[str, str]):
    rows: dict[str, dict] = {}  # pct_code -> row dict

    # 1. headings
    for kind, obj in stream:
        if kind != "heading":
            continue
        code = obj.code                       # "85.43"
        ch = code[:2]
        rows[code] = dict(
            pct_code=code, level="heading", parent_code=ch,
            description_raw=obj.desc, cd_percent=obj.cd, is_synthetic=0,
            _label_path=[],
        )

    # 2. leaves -> national, or heading-is-leaf when .0000
    sub_info = subheading_labels(stream)          # {sub: {"raw","path"}}
    for kind, obj in stream:
        if kind != "leaf":
            continue
        code = obj.code                       # "8543.7010"
        ch = code[:2]
        sub = code[:7]                         # "8543.70"
        desc, cd = extract_specific_duty(obj.desc, obj.cd)
        if code.endswith(".0000"):
            # heading-is-leaf: the 4-digit heading itself is the priced line
            rows[code] = dict(
                pct_code=code, level="heading", parent_code=ch,
                description_raw=desc, cd_percent=cd, is_synthetic=0,
                _label_path=[],
            )
            continue
        info = sub_info.get(sub, {"raw": "", "path": []})
        sub_chain = list(info["path"]) + ([info["raw"]] if info["raw"] else [])
        nat_path = national_label_path(obj, sub_chain)
        rows[code] = dict(
            pct_code=code, level="national", parent_code=sub,
            description_raw=desc, cd_percent=cd, is_synthetic=0,
            _label_path=nat_path,
        )

    # 3. synthesize subheading rows (6-digit) for every national-bearing group
    nationals = [r for r in rows.values() if r["level"] == "national"]
    for sub in sorted({r["parent_code"] for r in nationals}):
        head4 = sub[:2] + "." + sub[2:4]      # "85.43"
        info = sub_info.get(sub, {"raw": "", "path": []})
        rows[sub] = dict(
            pct_code=sub, level="subheading", parent_code=head4,
            description_raw=info["raw"], cd_percent=None, is_synthetic=1,
            _label_path=list(info["path"]),
        )

    # 4. synthesize chapter rows (2-digit) for every chapter present
    chapters_present = {r["pct_code"][:2] for r in rows.values()}
    for ch in sorted(chapters_present):
        rows[ch] = dict(
            pct_code=ch, level="chapter", parent_code=None,
            description_raw=chapter_titles.get(ch, ""), cd_percent=None,
            is_synthetic=1,
            _label_path=[],
        )

    # 5. ensure every referenced 4-digit heading exists (some headings only
    #    appear via leaves, never as a printed XX.XX row). Synthesize if missing.
    for r in list(rows.values()):
        p = r["parent_code"]
        if p and re.fullmatch(r"\d{2}\.\d{2}", p) and p not in rows:
            rows[p] = dict(
                pct_code=p, level="heading", parent_code=p[:2],
                description_raw=HEADING_TITLE_OVERRIDES.get(p, ""),
                cd_percent=None, is_synthetic=1,
                _label_path=[],
            )

    # 6. description_full = ancestor chain top->bottom, with each row's dash-label
    #    grouping path spliced in just above its own label.
    def full(code: str) -> str:
        segments: list[str] = []   # (top->bottom built reversed then flipped)
        cur = code
        guard = 0
        while cur is not None and guard < 8:
            r = rows.get(cur)
            if not r:
                break
            # own label first (we are walking bottom->top; flip later)
            if r["description_raw"]:
                segments.append(r["description_raw"])
            # then its label path, deepest-last, so when flipped they sit ABOVE
            for lbl in reversed(r.get("_label_path", [])):
                if lbl:
                    segments.append(lbl)
            cur = r["parent_code"]
            guard += 1
        segments.reverse()
        # collapse adjacent duplicates (single-leaf subheading == its leaf, or a
        # label repeated as the next row's raw)
        out = []
        for seg in segments:
            norm = seg.rstrip(":").strip().lower()
            if out and out[-1][1] == norm:
                continue
            out.append((seg, norm))
        return " > ".join(seg for seg, _ in out)

    for code, r in rows.items():
        r["description_full"] = full(code)
        r["fiscal_year"] = FISCAL_YEAR
        r.pop("_label_path", None)

    return rows


def write_db(rows: dict[str, dict], db_path: Path):
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS codes")
    con.execute(
        """
        CREATE TABLE codes (
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
    )
    con.executemany(
        """INSERT INTO codes
           (pct_code, level, parent_code, description_raw, description_full,
            cd_percent, fiscal_year, is_synthetic)
           VALUES (:pct_code, :level, :parent_code, :description_raw,
                   :description_full, :cd_percent, :fiscal_year, :is_synthetic)""",
        list(rows.values()),
    )
    con.commit()
    con.close()


def filter_stream_to_chapters(stream, chapters: set[str]):
    """Keep only stream items belonging to the given 2-digit chapters."""
    out = []
    keep = False
    for kind, obj in stream:
        if kind == "heading":
            keep = obj.code[:2] in chapters
        elif kind == "leaf":
            keep = obj.code[:2] in chapters
        # labels inherit the current keep state (they precede a leaf/heading of
        # the same chapter, or follow one)
        if keep:
            out.append((kind, obj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", help="comma-separated 2-digit chapters to limit to, e.g. 01,85")
    ap.add_argument("--pdf", default=str(PDF))
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()

    text = extract_text(Path(args.pdf))
    stream = parse_with_labels(text)
    if args.chapters:
        chs = {c.strip() for c in args.chapters.split(",")}
        stream = filter_stream_to_chapters(stream, chs)
    chapter_titles = load_chapter_titles(CHAPTER_TITLES_MD)
    rows = build_db(stream, chapter_titles)
    write_db(rows, Path(args.db))

    # summary
    con = sqlite3.connect(args.db)
    print("level counts:")
    for level, n in con.execute(
        "SELECT level, COUNT(*) FROM codes GROUP BY level ORDER BY 1"
    ):
        print(f"  {level:12s} {n}")
    orphans = con.execute(
        """SELECT COUNT(*) FROM codes WHERE parent_code IS NOT NULL
           AND parent_code NOT IN (SELECT pct_code FROM codes)"""
    ).fetchone()[0]
    print(f"orphan parent_code rows: {orphans}")
    con.close()


if __name__ == "__main__":
    main()
