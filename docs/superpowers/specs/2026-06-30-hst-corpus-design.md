# HST Corpus (US HS6) — Design

**Date:** 2026-06-30
**Scope:** Build a US Harmonized Tariff Schedule corpus **up to HS6 only**
(chapter / heading / subheading), scraped from Flexport, into a new
`hst/hst_corpus.db` with the **same `codes` schema** as the existing PCT corpus.
US-specific 8/10-digit national/statistical lines and duty rates are **deferred**
to a later effort. Embeddings (`vec_embeddings`) are a separate follow-up spec.

This replaces the PCT/Pakistan source (PDF) with the US HTS via the Flexport
website. The PCT pipeline is preserved (moved into a `pct/` folder), not deleted.

---

## 1. Source: Flexport HTS pages

- URL pattern: `https://tariffs.flexport.com/hscodes/us/<NN>/<slug>`
  (bare chapter / wrong slug → HTTP 403; the **exact slug is required**).
- Server-rendered HTML — codes present in the raw response, **no JS/headless
  browser needed** (`curl` retrieves everything).
- Covers all HS chapters. Chapter **77 is "reserved for possible future use"**
  (skip). Chapters 98/99 are US special/temporary provisions (include; they are
  valid HTS chapters).
- The full chapter→slug map (all chapters) is captured in
  `fetch_hts.py` as a hardcoded dict (sourced from the Flexport catalog index).

### HTML structure (verified against ch 01, 42, 85)

- **Chapter name:** `<h1 ...>Chapter <!-- -->NN<!-- -->: <!-- -->NAME</h1>`.
- **Heading:** anchor `<div ... id="hs-DDDD">` wrapping
  `<h2 class="subheading">DDDD<!-- -->: <!-- -->NAME</h2>`.
- **Code rows:** `<tr ... id="hs-<digits>">` where `<digits>` is the code with
  dots stripped. Lengths seen: 6 (bare HS6), 8, 10. First `<td>` holds the dotted
  code text; the description is in a `<div class="hidden md:block">NAME</div>`.
- **Grouping-label rows:** `<tr>` with an **empty first `<td>`** and a
  `colSpan="5"` cell `<div class="flex items-center"><div>LABEL</div></div>`.
  These carry NO code (e.g. `Cattle`, `Horses`, `Handbags…`, `Other DC motors…`).
  They are the dash-label grouping path — the same concept as PCT's synthesized
  grouping labels — and are NOT their own `codes` rows.

### Indentation is unreliable (verified, do not use)

Rows carry a visual indent via repeated `·` (`text-gray-300`) dot divs. Measured
on ch 42 and ch 85, the dot depth is **non-contiguous and does not map to code
level** (e.g. ch85 `8501.10` dots=0 but `8501.31` dots=1 at the same level;
8-digit and 10-digit rows share depths). This mirrors the PCT finding that dash
DEPTH is untrustworthy. **Level is derived from code-digit length only; dots are
ignored entirely.**

---

## 2. Output: `hst/hst_corpus.db`

`codes` table — **identical schema to `pct_corpus.db`** so the existing
`build_embeddings.py` can be repointed later with only a path change:

```sql
CREATE TABLE codes (
    pct_code         TEXT PRIMARY KEY,   -- column name kept for embeddings compat
    level            TEXT,               -- 'chapter' | 'heading' | 'subheading'
    parent_code      TEXT,               -- FK self
    description_raw  TEXT,               -- own label (may be empty for some HS6)
    description_full TEXT,               -- chapter > heading > [labels…] > own
    cd_percent       TEXT,               -- NULL (duty deferred)
    fiscal_year      TEXT,               -- '2026'
    is_synthetic     BOOLEAN
)
```

### Levels (HS6 ceiling — no `national` level)

| level | `pct_code` format | source |
|---|---|---|
| `chapter` | `01` | `<h1>Chapter 01: …</h1>` |
| `heading` | `0101` | `<h2>0101: …</h2>` (`id="hs-0101"`) |
| `subheading` | `0101.21` | code anchor ID, first 6 digits, dotted `XXXX.XX` |

Code format: chapter = 2-digit undotted; heading = 4-digit undotted; subheading =
`XXXX.XX`. (`pct_code` is an opaque PK string; embeddings never parse it.)

---

## 3. Parse rules (per chapter HTML)

Source of truth for codes/levels = the **`id="hs-<digits>"` anchors** (cleaner
than the visible dotted cell text, which varies). Names + grouping labels come
from the visible cells (not present in IDs). Parsing is done with regex /
string-split over the known, stable React markup (no bs4/lxml dependency).

1. **Chapter row** — from `<h1>`. `pct_code=NN`, level=`chapter`, parent=NULL,
   `description_raw`=`description_full`=NAME, `is_synthetic=1` (it is a page
   header, not a table row).

2. **Heading rows** — each `<h2 class="subheading">` / `id="hs-DDDD"`.
   `pct_code=DDDD`, parent=`NN`, raw=NAME, full=`<chapter> > NAME`,
   `is_synthetic=0`.

3. **Walk table rows in DOM order** with a **flat label buffer**:
   - **LABEL row** (colSpan, no id) → append its text to the label buffer.
   - **CODE row** (`id="hs-<digits>"`):
     - Compute HS6 = first 6 digits → dotted `XXXX.XX`.
     - **First time** this HS6 is seen → emit a subheading row:
       - `parent_code` = the 4-digit heading.
       - `description_raw` = the row's own name div.
       - `description_full` = `<chapter> > <heading> > [buffered labels in order] > raw`.
       - `is_synthetic` = 0 if a bare 6-digit anchor (`id="hs-DDDDDD"`) exists for
         this HS6 in the HTML; 1 if the HS6 is only derived from longer
         (8/10-digit) children.
       - **Clear the label buffer** after emitting.
     - HS6 **already emitted** (row is an 8/10-digit child of a seen subheading)
       → **skip** (HS6 ceiling; national lines deferred).

4. **Grouping-label attach is flat** (single-pass buffer; no nested label tree).
   Labels accumulated since the previous emitted HS6 are spliced, in DOM order,
   into the next subheading's `description_full`. Deep label nesting is not
   modeled — at HS6 granularity it is rare, and the dot-based nesting hint is
   unreliable (§1). This is the PCT "splice labels in order" rule.

5. **Edge cases (the cattle/horses case):**
   - `0102.29` prints a bare HS6 row (`id="hs-010229"`, name "Other") →
     `raw="Other"`, `is_synthetic=0`.
   - `0102.21` appears only as `0102.21.00` (`id="hs-01022100"`) with no bare
     6-digit row → synthesize HS6 `0102.21`, `raw` from the 8-digit row's name
     ("Purebred breeding animals"), `is_synthetic=1`.
   - Grouping label "Cattle" (colSpan) → buffered → spliced into the
     `description_full` of `0102.21` / `0102.29`; it is NOT its own row.

6. **Empty `description_raw`** is allowed: a synthesized HS6 whose longer
   children diverge may have no single own label (same as PCT data-limit #2). The
   discriminating grouping labels still live in `description_full`.

---

## 4. Architecture: two-phase, cached

All new code lives under `hst/`.

| Path | Role |
|---|---|
| `hst/fetch_hts.py` | **Phase 1 (network).** Loop chapters 01–97 (skip 77) using the hardcoded slug map. One GET per chapter → save raw HTML to `hst/hts_cache/<NN>.html`. Skip-if-cached; `--force` refetch; `--chapters 42,85` filter; polite ~1s delay; browser User-Agent. |
| `hst/hts_cache/` | Raw chapter HTML cache (runtime; gitignore-worthy). |
| `hst/build_hst_codes.py` | **Phase 2 (offline parse).** Parse `hst/hts_cache/*.html` → `hst/hst_corpus.db` `codes` table. No network. `--chapters` filter. Prints a **per-chapter summary** (rows per level + any warnings) so the one-time build can be monitored/steered. Re-runnable instantly without refetching. |
| `hst/hst_corpus.db` | Output SQLite. |

Network is confined to Phase 1; Phase 2 is pure parsing so a parse-rule tweak
never triggers a refetch.

### Steering / monitoring

- Per-chapter summary line: `ch NN  <name>  headings=H subheadings=S (synth=X)  [warnings]`.
- Warnings to surface: 0 subheadings in a chapter, heading with no subheadings,
  HS6 with empty raw AND no labels, HTTP non-200 in fetch.
- `--chapters` on both scripts to fetch/parse a single chapter for spot-checking.

---

## 5. Repo reorg

Move existing PCT pipeline into `pct/`:

| From | To |
|---|---|
| `build_codes.py` | `pct/build_codes.py` |
| `build_embeddings.py` | `pct/build_embeddings.py` |
| `pct_corpus.db` | `pct/pct_corpus.db` |
| `PTC-2025-26.pdf` | `pct/` *(file currently missing on disk — move if present, else just fix the stale reference)* |

**Stays at repo root** (shared / VLM, not PCT-specific): `serve.sh`,
`test_infer.py`, `test_infer_dense.py`, `model.py` (if present), `docs/`,
`images/`, `file`, `vllm.log`, `tests/`.

### Path reroutes after the move

- `pct/build_codes.py`, `pct/build_embeddings.py`: use `HERE = __file__.parent`,
  so the db/pdf move alongside them — **no code change needed**.
- `test_infer.py:33`, `test_infer_dense.py:33`:
  `DB = __file__.parent / "pct_corpus.db"` → change to
  `__file__.parent / "pct" / "pct_corpus.db"`.
- `tests/test_dash_labels.py`: imports `build_codes` via repo root on
  `sys.path` → change the inserted path to `parent.parent / "pct"`.
- `CLAUDE.md` + `docs/*.md`: update stale path text (`build_codes.py`,
  `pct_corpus.db`, etc.) to the new `pct/` locations; document the new `hst/`
  pipeline.

---

## 6. Out of scope (explicit)

- US 8/10-digit national & statistical lines (the last 4 digits) — later.
- Duty rates (`cd_percent` stays NULL) — later, with the national lines.
- Embeddings / `vec_embeddings` for HST — separate follow-up spec. It will reuse
  `build_embeddings.py` repointed at `hst_corpus.db`, with the same parent-discard
  `description_own` rule (chapter=`segs[0]`, heading=`segs[1:]`,
  subheading=`segs[2:]`).

---

## 7. Verification

- Spot-check ch 01: expect headings `0101`–`0106`, HS6 incl. `0101.21`
  (synthetic, "Purebred breeding animals"), `0102.29` (real, "Other"),
  grouping label "Cattle" present in `0102.21`/`0102.29` `description_full`.
- Spot-check ch 42: `4202.21`/`4202.22`/`4202.31` etc. with the
  "Handbags…", "Trunks, suitcases…" grouping labels spliced into full desc and
  NOT collapsing to identical sibling text.
- Spot-check ch 85: motors `8501.xx` subheadings present; no 8/10-digit rows
  leaked into the table.
- Sanity: every `subheading.parent_code` exists as a `heading` row; every
  `heading.parent_code` exists as a `chapter` row; no `level='national'` rows.
