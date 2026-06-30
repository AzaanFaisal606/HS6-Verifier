# HST Corpus (US HS6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scrape the US Harmonized Tariff Schedule (chapter / heading / HS6 subheading only) from Flexport into a new `hst/hst_corpus.db` with the same `codes` schema as the existing PCT corpus, and reorganize the PCT pipeline into a `pct/` folder.

**Architecture:** Two-phase. `hst/fetch_hts.py` downloads each chapter's raw HTML to `hst/hts_cache/<NN>.html` (network only here). `hst/build_hst_codes.py` parses the cache offline into `hst/hst_corpus.db`. Codes/levels come from `id="hs-<digits>"` anchors; names + grouping labels from visible cells; indentation dots are ignored (proven unreliable). Existing PCT files move into `pct/`.

**Tech Stack:** Python 3 (conda env `vision`), stdlib only — `urllib`/`curl`, `re`, `html`, `sqlite3`, `pathlib`, `argparse`. No bs4/lxml. No pytest (repo uses standalone no-pytest test scripts run via `python`).

## Global Constraints

- Run all project code in the `vision` conda env: `~/miniconda3/envs/vision/bin/python` (or `source ~/miniconda3/etc/profile.d/conda.sh && conda activate vision`).
- HS6 ceiling: emit only `chapter` / `heading` / `subheading` rows. Never emit `national` rows or any 8/10-digit code.
- `codes` schema is **byte-identical** to `pct_corpus.db` (PK column stays named `pct_code`).
- `pct_code` formats: chapter `01` (2-digit), heading `0101` (4-digit), subheading `0101.21` (`XXXX.XX`).
- `cd_percent` = NULL (duty deferred). `fiscal_year` = `'2026'`.
- Level is derived from code-digit length only. **Ignore the `·` indentation dots entirely.**
- Grouping-label attach is flat (single buffer, DOM order); no nested label tree.
- Repo is NOT a git repository — there are no commit steps. Each task ends with a verification command + expected output instead.
- Skip chapter 77 (reserved). Include chapters 98 and 99.
- Flexport requires the exact `<slug>`; a bare/wrong slug returns HTTP 403. Send a browser `User-Agent`.

---

### Task 1: Repo reorg into `pct/` + path reroutes

Move the PCT pipeline into `pct/` and fix every broken path. No new logic.

**Files:**
- Create dir: `pct/`
- Move: `build_codes.py` → `pct/build_codes.py`
- Move: `build_embeddings.py` → `pct/build_embeddings.py`
- Move: `pct_corpus.db` → `pct/pct_corpus.db`
- Move (if present): `PTC-2025-26.pdf` → `pct/PTC-2025-26.pdf`
- Modify: `test_infer.py` (DB path)
- Modify: `test_infer_dense.py` (DB path)
- Modify: `tests/test_dash_labels.py` (import path)

**Interfaces:**
- Produces: `pct/pct_corpus.db` reachable from root scripts at `Path(__file__).resolve().parent / "pct" / "pct_corpus.db"`.

- [ ] **Step 1: Create the folder and move files**

```bash
cd /home/azaan/Vision
mkdir -p pct
mv build_codes.py build_embeddings.py pct_corpus.db pct/
[ -f PTC-2025-26.pdf ] && mv PTC-2025-26.pdf pct/ || echo "PTC pdf not present, skipping"
```

- [ ] **Step 2: Verify the moved scripts still resolve their own paths**

`pct/build_codes.py` and `pct/build_embeddings.py` use `HERE = Path(__file__).resolve().parent`, so the db/pdf moved alongside them — no edit needed. Confirm:

Run: `cd /home/azaan/Vision && grep -n "HERE = " pct/build_codes.py pct/build_embeddings.py`
Expected: each prints a line assigning `HERE` from `Path(__file__).resolve().parent`.

- [ ] **Step 3: Reroute `test_infer.py` DB path**

In `test_infer.py`, change line 33:

```python
DB = Path(__file__).resolve().parent / "pct" / "pct_corpus.db"
```
(was `... .parent / "pct_corpus.db"`)

- [ ] **Step 4: Reroute `test_infer_dense.py` DB path**

In `test_infer_dense.py`, change line 33:

```python
DB = Path(__file__).resolve().parent / "pct" / "pct_corpus.db"
```

- [ ] **Step 5: Reroute the test import path**

In `tests/test_dash_labels.py`, change line 6 so `build_codes` is importable from its new home:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pct"))  # pct/ for build_codes
```
(was `... .parent.parent`)

- [ ] **Step 6: Verify the existing PCT tests still pass after the move**

Run: `cd /home/azaan/Vision && ~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: the test script runs to completion with its existing PASS output (no `ModuleNotFoundError: build_codes`).

- [ ] **Step 7: Verify a PCT corpus query still works through the new path**

Run: `cd /home/azaan/Vision && ~/miniconda3/envs/vision/bin/python -c "import sqlite3,pathlib; db=pathlib.Path('pct/pct_corpus.db'); print(db.exists(), sqlite3.connect(db).execute('select count(*) from codes').fetchone())"`
Expected: `True (14107,)`

---

### Task 2: Chapter slug map + `fetch_hts.py` (Phase 1)

Download each chapter's raw HTML to the cache.

**Files:**
- Create: `hst/fetch_hts.py`
- Create dir (at runtime): `hst/hts_cache/`

**Interfaces:**
- Produces: `CHAPTERS: dict[str, str]` mapping 2-digit chapter → full URL path slug (e.g. `"01": "live-animals"`). Used by Task 3 only for the human-readable summary fallback; Task 3 reads cached HTML, not this map.
- Produces: cache files `hst/hts_cache/<NN>.html` consumed by Task 3.

- [ ] **Step 1: Write `hst/fetch_hts.py` with the full slug map and fetch loop**

```python
#!/usr/bin/env python3
"""Phase 1: download US HTS chapter pages from Flexport into hst/hts_cache/.

Network only. Re-run is a no-op for already-cached chapters unless --force.
Usage:
  python fetch_hts.py                 # fetch all (skip cached)
  python fetch_hts.py --chapters 01,42,85
  python fetch_hts.py --force         # refetch even if cached
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "hts_cache"
BASE = "https://tariffs.flexport.com/hscodes/us"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DELAY_S = 1.0

# Chapter -> URL slug. Chapter 77 reserved (no page). Sourced from the
# Flexport /hscodes catalog index on 2026-06-30.
CHAPTERS = {
    "01": "live-animals",
    "02": "meat-and-edible-meat-offal",
    "03": "fish-and-crustaceans-molluscs-and-other-aquatic-invertebrates",
    "04": "dairy-produce-birds-eggs-natural-honey-edible-products-of-animal",
    "05": "products-of-animal-origin-not-elsewhere-specified-or-included",
    "06": "live-trees-and-other-plants-bulbs-roots-and-the-like",
    "07": "edible-vegetables-and-certain-roots-and-tubers",
    "08": "edible-fruit-and-nuts-peel-of-citrus-fruit-or-melons",
    "09": "coffee-tea-mat-and-spices",
    "10": "cereals",
    "11": "products-of-the-milling-industry-malt-starches-inulin-wheat-gluten",
    "12": "oil-seeds-and-oleaginous-fruits-miscellaneous-grains-seeds-and-fruits",
    "13": "lac-gums-resins-and-other-vegetable-saps-and-extracts",
    "14": "vegetable-plaiting-materials-vegetable-products-not-elsewhere-specified-or-included",
    "15": "animal-or-vegetable-fats-and-oils-and-their-cleavage-products",
    "16": "preparations-of-meat-of-fish-or-of-crustaceans-molluscs-or",
    "17": "sugars-and-sugar-confectionery",
    "18": "cocoa-and-cocoa-preparations",
    "19": "preparations-of-cereals-flour-starch-or-milk-bakers-wares",
    "20": "preparations-of-vegetables-fruit-nuts-or-other-parts-of-plants",
    "21": "miscellaneous-edible-preparations",
    "22": "beverages-spirits-and-vinegar",
    "23": "residues-and-waste-from-the-food-industries-prepared-animal-feed",
    "24": "tobacco-and-manufactured-tobacco-substitutes",
    "25": "salt-sulfur-earths-and-stone-plastering-materials-lime-and-cement",
    "26": "ores-slag-and-ash",
    "27": "mineral-fuels-mineral-oils-and-products-of-their-distillation-bituminous",
    "28": "inorganic-chemicals-organic-or-inorganic-compounds-of-precious-metals-of",
    "29": "organic-chemicals",
    "30": "pharmaceutical-products",
    "31": "fertilizers",
    "32": "tanning-or-dyeing-extracts-dyes-pigments-paints-varnishes-putty-and",
    "33": "essential-oils-and-resinoids-perfumery-cosmetic-or-toilet-preparations",
    "34": "soap-organic-surface-active-agents-washing-preparations-lubricating-preparations-artificial",
    "35": "albuminoidal-substances-modified-starches-glues-enzymes",
    "36": "explosives-pyrotechnic-products-matches-pyrophoric-alloys-certain-combustible-preparations",
    "37": "photographic-or-cinematographic-goods",
    "38": "miscellaneous-chemical-products",
    "39": "plastics-and-articles-thereof",
    "40": "rubber-and-articles-thereof",
    "41": "raw-hides-and-skins-other-than-furskins-and-leather",
    "42": "articles-of-leather-saddlery-and-harness-travel-goods-handbags-and",
    "43": "furskins-and-artificial-fur-manufactures-thereof",
    "44": "wood-and-articles-of-wood-wood-charcoal",
    "45": "cork-and-articles-of-cork",
    "46": "manufactures-of-straw-of-esparto-or-of-other-plaiting-materials",
    "47": "pulp-of-wood-or-of-other-fibrous-cellulosic-material-waste",
    "48": "paper-and-paperboard-articles-of-paper-pulp-of-paper-or",
    "49": "printed-books-newspapers-pictures-and-other-products-of-the-printing",
    "50": "silk",
    "51": "wool-fine-or-coarse-animal-hair-horsehair-yarn-and-woven",
    "52": "cotton",
    "53": "other-vegetable-textile-fibers-paper-yarn-and-woven-fabric-of",
    "54": "man-made-filaments",
    "55": "man-made-staple-fibers",
    "56": "wadding-felt-and-nonwovens-special-yarns-twine-cordage-ropes-and",
    "57": "carpets-and-other-textile-floor-coverings",
    "58": "special-woven-fabrics-tufted-textile-fabrics-lace-tapestries-trimmings-embroidery",
    "59": "impregnated-coated-covered-or-laminated-textile-fabrics-textile-articles-of",
    "60": "knitted-or-crocheted-fabrics",
    "61": "articles-of-apparel-and-clothing-accessories-knitted-or-crocheted",
    "62": "articles-of-apparel-and-clothing-accessories-not-knitted-or-crocheted",
    "63": "other-made-up-textile-articles-sets-worn-clothing-and-worn",
    "64": "footwear-gaiters-and-the-like-parts-of-such-articles",
    "65": "headgear-and-parts-thereof",
    "66": "umbrellas-sun-umbrellas-walking-sticks-seatsticks-whips-riding-crops-and",
    "67": "prepared-feathers-and-down-and-articles-made-of-feathers-or",
    "68": "articles-of-stone-plaster-cement-asbestos-mica-or-similar-materials",
    "69": "ceramic-products",
    "70": "glass-and-glassware",
    "71": "natural-or-cultured-pearls-precious-or-semi-precious-stones-precious",
    "72": "iron-and-steel",
    "73": "articles-of-iron-or-steel",
    "74": "copper-and-articles-thereof",
    "75": "nickel-and-articles-thereof",
    "76": "aluminum-and-articles-thereof",
    # 77 reserved for possible future use - no page
    "78": "lead-and-articles-thereof",
    "79": "zinc-and-articles-thereof",
    "80": "tin-and-articles-thereof",
    "81": "other-base-metals-cermets-articles-thereof",
    "82": "tools-implements-cutlery-spoons-and-forks-of-base-metal-parts",
    "83": "miscellaneous-articles-of-base-metal",
    "84": "nuclear-reactors-boilers-machinery-and-mechanical-appliances-parts-thereof",
    "85": "electrical-machinery-and-equipment-and-parts-thereof-sound-recorders-and",
    "86": "railway-or-tramway-locomotives-rolling-stock-and-parts-thereof-railway",
    "87": "vehicles-other-than-railway-or-tramway-rolling-stock-and-parts",
    "88": "aircraft-spacecraft-and-parts-thereof",
    "89": "ships-boats-and-floating-structures",
    "90": "optical-photographic-cinematographic-measuring-checking-precision-medical-or-surgical-instruments",
    "91": "clocks-and-watches-and-parts-thereof",
    "92": "musical-instruments-parts-and-accessories-of-such-articles",
    "93": "arms-and-ammunition-parts-and-accessories-thereof",
    "94": "furniture-bedding-mattresses-mattress-supports-cushions-and-similar-stuffed-furnishings",
    "95": "toys-games-and-sports-requisites-parts-and-accessories-thereof",
    "96": "miscellaneous-manufactured-articles",
    "97": "works-of-art-collectors-pieces-and-antiques",
    "98": "special-classification-provisions",
    "99": "temporary-legislation-temporary-modifications-proclaimed-pursuant-to-trade-agreements-legislation",
}


def fetch_one(ch: str, slug: str, force: bool) -> str:
    out = CACHE / f"{ch}.html"
    if out.exists() and not force:
        return "cached"
    url = f"{BASE}/{ch}/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    out.write_bytes(body)
    return f"OK {len(body)} bytes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", help="comma list e.g. 01,42,85")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    if args.chapters:
        want = [c.strip().zfill(2) for c in args.chapters.split(",")]
    else:
        want = list(CHAPTERS)

    rc = 0
    for ch in want:
        slug = CHAPTERS.get(ch)
        if not slug:
            print(f"ch {ch}: SKIP (no slug / reserved)")
            continue
        try:
            status = fetch_one(ch, slug, args.force)
            print(f"ch {ch}: {status}")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"ch {ch}: ERROR {e}")
            rc = 1
        if status != "cached":  # type: ignore[name-defined]
            time.sleep(DELAY_S)
    return rc


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Fetch chapter 01 only and verify the cache file**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python fetch_hts.py --chapters 01`
Expected: prints `ch 01: OK <N> bytes`; file `hst/hts_cache/01.html` exists and is > 100 KB.

- [ ] **Step 3: Verify re-run is a cached no-op**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python fetch_hts.py --chapters 01`
Expected: prints `ch 01: cached`.

---

### Task 3: HTML parser — `parse_chapter()` (pure, tested with fixtures)

Pure parsing function: HTML string → ordered list of row dicts. This is the testable core. No DB, no file IO.

**Files:**
- Create: `hst/build_hst_codes.py` (parser portion only this task)
- Create: `hst/tests/test_parse.py`
- Create: `hst/tests/fixtures.py`

**Interfaces:**
- Produces: `parse_chapter(html: str) -> list[dict]`. Each dict has keys:
  `pct_code: str`, `level: str` (`'chapter'|'heading'|'subheading'`),
  `parent_code: str | None`, `description_raw: str`, `description_full: str`,
  `is_synthetic: bool`. Rows are returned in emission order (chapter, then
  per-heading: heading row then its subheadings). Consumed by Task 4.

- [ ] **Step 1: Capture real HTML fixtures from the cache**

Requires Task 2's `hst/hts_cache/01.html`. Create `hst/tests/fixtures.py` by extracting small verbatim slices. Generate it with this helper (run once, then hand-trim is unnecessary — it writes the file):

```bash
cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python - <<'PY'
from pathlib import Path
html = (Path("hts_cache") / "01.html").read_text()
# Whole chapter-1 HTML is small enough to embed as a fixture verbatim.
out = Path("tests"); out.mkdir(exist_ok=True)
(out / "fixtures.py").write_text(
    "# Verbatim Flexport chapter-01 HTML, captured 2026-06-30 for parser tests.\n"
    "CH01_HTML = r'''" + html.replace("'''", "'' '") + "'''\n"
)
print("wrote tests/fixtures.py", len((out / 'fixtures.py').read_text()), "chars")
PY
```

Expected: prints `wrote tests/fixtures.py <N> chars`.

- [ ] **Step 2: Write the failing parser test**

Create `hst/tests/test_parse.py`:

```python
"""Standalone (no-pytest) tests for the HST chapter parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # hst/ for build_hst_codes
import fixtures as F  # noqa: E402
import build_hst_codes as B  # noqa: E402


def by_code(rows):
    return {r["pct_code"]: r for r in rows}


def test_chapter_row():
    rows = B.parse_chapter(F.CH01_HTML)
    d = by_code(rows)
    assert d["01"]["level"] == "chapter"
    assert d["01"]["description_raw"] == "Live animals"
    assert d["01"]["parent_code"] is None


def test_heading_rows():
    d = by_code(B.parse_chapter(F.CH01_HTML))
    assert set(["0101", "0102", "0103", "0104", "0105", "0106"]) <= set(d)
    assert d["0101"]["level"] == "heading"
    assert d["0101"]["parent_code"] == "01"
    assert d["0101"]["description_raw"] == "Live horses, asses, mules and hinnies"


def test_real_hs6_with_bare_code():
    # 0102.29 prints a bare 6-digit row, name "Other", not synthetic.
    d = by_code(B.parse_chapter(F.CH01_HTML))
    assert d["0102.29"]["level"] == "subheading"
    assert d["0102.29"]["parent_code"] == "0102"
    assert d["0102.29"]["description_raw"] == "Other"
    assert d["0102.29"]["is_synthetic"] is False


def test_synthesized_hs6_from_eight_digit():
    # 0102.21 appears only as 0102.21.00 -> synthesized HS6.
    d = by_code(B.parse_chapter(F.CH01_HTML))
    assert d["0102.21"]["level"] == "subheading"
    assert d["0102.21"]["is_synthetic"] is True
    assert d["0102.21"]["description_raw"] == "Purebred breeding animals"


def test_grouping_label_in_full_desc():
    # "Cattle" colSpan label must be spliced into 0102.21 / 0102.29 full desc,
    # and must NOT exist as its own row.
    rows = B.parse_chapter(F.CH01_HTML)
    d = by_code(rows)
    assert "Cattle" in d["0102.21"]["description_full"]
    assert d["0102.21"]["description_full"].startswith(
        "Live animals > Live bovine animals"
    )
    assert all("Cattle" != r["pct_code"] for r in rows)


def test_no_national_rows():
    rows = B.parse_chapter(F.CH01_HTML)
    assert all(r["level"] in ("chapter", "heading", "subheading") for r in rows)
    # no 8/10-digit codes leaked
    assert all(len(r["pct_code"].replace(".", "")) <= 6 for r in rows)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python tests/test_parse.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_hst_codes'` (or `AttributeError: ... parse_chapter`).

- [ ] **Step 4: Implement `parse_chapter` in `hst/build_hst_codes.py`**

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python tests/test_parse.py`
Expected: `PASS test_chapter_row` … through all tests, ending `6 passed`.

---

### Task 4: DB writer + CLI + per-chapter summary

Wrap the parser: read cache, build the table, print per-chapter summaries, emit warnings.

**Files:**
- Modify: `hst/build_hst_codes.py` (add `build_db`, `summarize`, `main`)
- Create: `hst/tests/test_build_db.py`

**Interfaces:**
- Consumes: `parse_chapter(html) -> list[dict]` from Task 3.
- Produces: `build_db(rows: list[dict], db_path: Path) -> None` (creates `codes` table if absent, inserts/replaces rows). `summarize(ch: str, rows: list[dict]) -> str` (one summary line). `main()` CLI with `--chapters`.

- [ ] **Step 1: Write the failing DB test**

Create `hst/tests/test_build_db.py`:

```python
"""Standalone test: parse_chapter -> build_db round-trip on chapter 01."""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fixtures as F  # noqa: E402
import build_hst_codes as B  # noqa: E402


def test_build_db_roundtrip():
    rows = B.parse_chapter(F.CH01_HTML)
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        B.build_db(rows, db)
        con = sqlite3.connect(db)
        # schema columns match PCT exactly
        cols = [r[1] for r in con.execute("PRAGMA table_info(codes)")]
        assert cols == [
            "pct_code", "level", "parent_code", "description_raw",
            "description_full", "cd_percent", "fiscal_year", "is_synthetic",
        ]
        n_chap = con.execute(
            "select count(*) from codes where level='chapter'"
        ).fetchone()[0]
        n_head = con.execute(
            "select count(*) from codes where level='heading'"
        ).fetchone()[0]
        assert n_chap == 1 and n_head == 6
        # cd_percent NULL, fiscal_year set
        fy, cd = con.execute(
            "select fiscal_year, cd_percent from codes where pct_code='0102.29'"
        ).fetchone()
        assert fy == "2026" and cd is None
        # referential sanity: every subheading parent is a heading
        bad = con.execute(
            "select count(*) from codes s where s.level='subheading' and "
            "s.parent_code not in (select pct_code from codes where level='heading')"
        ).fetchone()[0]
        assert bad == 0


if __name__ == "__main__":
    test_build_db_roundtrip()
    print("PASS test_build_db_roundtrip\n\n1 passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python tests/test_build_db.py`
Expected: FAIL — `AttributeError: module 'build_hst_codes' has no attribute 'build_db'`.

- [ ] **Step 3: Append `build_db`, `summarize`, and `main` to `hst/build_hst_codes.py`**

```python
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
```

- [ ] **Step 4: Run the DB test to verify it passes**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python tests/test_build_db.py`
Expected: `PASS test_build_db_roundtrip` then `1 passed`.

- [ ] **Step 5: Run the CLI on chapter 01 end-to-end**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python build_hst_codes.py --chapters 01`
Expected: a summary line `ch 01  Live animals ... headings=6 subheadings=N (synth=M)` then `wrote <N> rows to hst_corpus.db`; file `hst/hst_corpus.db` exists.

---

### Task 5: Full fetch + full build + corpus validation

Run the whole pipeline across all chapters and validate the result.

**Files:**
- Create: `hst/tests/validate_corpus.py`
- Uses: `hst/fetch_hts.py`, `hst/build_hst_codes.py`

**Interfaces:**
- Consumes: a fully built `hst/hst_corpus.db`.

- [ ] **Step 1: Fetch all chapters**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python fetch_hts.py`
Expected: one line per chapter, each `OK <bytes>` or `cached`; `ch 77: SKIP`; no `ERROR` lines. (If any `ERROR`, re-run `--chapters <NN>` for the failures before proceeding.)

- [ ] **Step 2: Build the full corpus and eyeball the per-chapter summaries**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python build_hst_codes.py`
Expected: ~96 summary lines, each with `headings>0` and `subheadings>0`; investigate any `NO-SUBHEADINGS` or `empty-heading` warning before accepting. Final `wrote <N> rows to hst_corpus.db`.

- [ ] **Step 3: Write the corpus validator**

Create `hst/tests/validate_corpus.py`:

```python
"""Validate the built hst_corpus.db: structure, FKs, HS6 ceiling, spot checks."""
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "hst_corpus.db"


def main() -> int:
    con = sqlite3.connect(DB)
    q = con.execute
    fails = []

    # no national / no >6-digit codes
    leaked = q(
        "select count(*) from codes where level not in "
        "('chapter','heading','subheading')"
    ).fetchone()[0]
    if leaked:
        fails.append(f"{leaked} non-HS6 level rows")

    # every heading has a valid chapter parent
    bad_h = q(
        "select count(*) from codes h where h.level='heading' and h.parent_code "
        "not in (select pct_code from codes where level='chapter')"
    ).fetchone()[0]
    if bad_h:
        fails.append(f"{bad_h} headings with bad parent")

    # every subheading has a valid heading parent
    bad_s = q(
        "select count(*) from codes s where s.level='subheading' and "
        "s.parent_code not in (select pct_code from codes where level='heading')"
    ).fetchone()[0]
    if bad_s:
        fails.append(f"{bad_s} subheadings with bad parent")

    # cd_percent all NULL; fiscal_year all 2026
    if q("select count(*) from codes where cd_percent is not null").fetchone()[0]:
        fails.append("non-null cd_percent present")
    if q("select count(*) from codes where fiscal_year<>'2026'").fetchone()[0]:
        fails.append("fiscal_year != 2026 present")

    # chapter-01 spot checks
    def raw(code):
        r = q("select description_raw from codes where pct_code=?", (code,)).fetchone()
        return r[0] if r else None

    def full(code):
        r = q("select description_full from codes where pct_code=?", (code,)).fetchone()
        return r[0] if r else None

    if raw("0102.29") != "Other":
        fails.append("0102.29 raw != 'Other'")
    if raw("0102.21") != "Purebred breeding animals":
        fails.append("0102.21 raw mismatch")
    if "Cattle" not in (full("0102.21") or ""):
        fails.append("Cattle label missing from 0102.21 full desc")

    counts = dict(
        q("select level, count(*) from codes group by level").fetchall()
    )
    print("level counts:", counts)
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("OK - all corpus validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the validator**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python tests/validate_corpus.py`
Expected: prints `level counts: {'chapter': 96..97, 'heading': ~1200+, 'subheading': ~5000+}` then `OK - all corpus validations passed`.

- [ ] **Step 5: Spot-check chapter 42 anti-collapse (the PCT bug regression)**

Run: `cd /home/azaan/Vision/hst && ~/miniconda3/envs/vision/bin/python -c "import sqlite3; c=sqlite3.connect('hst_corpus.db'); [print(r) for r in c.execute(\"select pct_code, description_full from codes where pct_code in ('4202.21','4202.22','4202.31') order by pct_code\")]"`
Expected: three rows with **distinct** `description_full` strings (each carrying its own grouping label like "Handbags…"), not byte-identical.

---

### Task 6: Update docs (CLAUDE.md + layout table)

Reflect the reorg and the new HST pipeline.

**Files:**
- Modify: `CLAUDE.md`
- Modify (path text only): `docs/sample.md`, `docs/remote-gpu-setup.md`, `docs/issues.md` where they name moved PCT files.

**Interfaces:** none (docs).

- [ ] **Step 1: Update the `CLAUDE.md` Layout table**

Add `pct/` prefix to `build_codes.py`, `build_embeddings.py`, `pct_corpus.db`, `PTC-2025-26.pdf`. Add new rows for `hst/fetch_hts.py`, `hst/build_hst_codes.py`, `hst/hst_corpus.db`, `hst/hts_cache/`. Add a short "US HST corpus (HS6)" subsection pointing at `docs/superpowers/specs/2026-06-30-hst-corpus-design.md`, noting: HS6 ceiling, anchor-ID source, flat labels, dots ignored, duty/national deferred.

- [ ] **Step 2: Fix stale path references in docs**

In `docs/sample.md`, `docs/remote-gpu-setup.md`, `docs/issues.md`, update any literal `build_codes.py` / `build_embeddings.py` / `pct_corpus.db` / `PTC-2025-26.pdf` that refer to the working-tree location to the new `pct/...` paths. (Leave `/home/azaan/Documents/PTC Corpus/...` external paths in `remote-gpu-setup.md` alone — those are a different machine's layout.)

- [ ] **Step 3: Verify no stale root-relative references remain**

Run: `cd /home/azaan/Vision && grep -rnE "(^|[^/])(build_codes\.py|build_embeddings\.py|pct_corpus\.db)" CLAUDE.md docs/*.md | grep -v "pct/" | grep -v "Documents/PTC"`
Expected: no output (every working-tree reference now points at `pct/`), or only intentional prose mentions.

---

## Self-Review

**Spec coverage:** §1 source/structure → Tasks 2,3. §1 indentation-ignored → Task 3 parser (`seen6`/digit-length, no dot use). §2 schema → Task 4 `_SCHEMA` + test asserting exact columns. §2 levels/formats → Task 3 `_hs6`, heading/chapter regex. §3 parse rules incl. cattle/horses, synthetic flag, flat labels, empty raw → Task 3 + its 6 tests. §4 two-phase/cache/CLI/summary/steering → Tasks 2,4. §5 reorg + reroutes → Task 1. §6 out-of-scope (NULL duty, no national, embeddings later) → enforced in Task 4 writer + Task 5 validator. §7 verification → Task 5 validator + ch42 spot check.

**Placeholder scan:** No TBD/TODO; all code shown in full; all commands have expected output.

**Type consistency:** `parse_chapter(html)->list[dict]` with the six documented keys is produced in Task 3 and consumed identically in Tasks 4–5. `build_db(rows, db_path)`, `summarize(ch, rows)` signatures match across Task 4 definition and tests. `is_synthetic` is a bool in dicts, written as `1/0` int in `build_db` (test checks via fiscal/cd, not the bool encoding — consistent).

**Note on emission order vs `seen6`:** `seen6` is scoped per-heading section, so an HS6 cannot be double-emitted within a heading; HS6 codes never span headings (first 4 digits = heading), so per-section scoping is correct.
