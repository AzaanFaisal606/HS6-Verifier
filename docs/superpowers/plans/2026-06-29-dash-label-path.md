# Dash-Label Path Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `build_codes.py` so `description_full` includes the intermediate dash-label grouping lines, making sibling subheadings/national lines textually distinct instead of byte-identical.

**Architecture:** Keep the existing dash-stack snapshot in `parse_with_labels`. Replace the single-label subheading heuristic with a shared-prefix-of-group rule, derive an above-HS6 subheading label path and a below-HS6 national label path per code, and splice both into `description_full`. No schema change.

**Tech Stack:** Python 3 (stdlib only: `re`, `sqlite3`, `subprocess`), `pdftotext` from poppler (only in the `vision` conda env), SQLite.

## Global Constraints

- Run all project code in the `vision` conda env. Binary: `~/miniconda3/envs/vision/bin/python`. `pdftotext` only exists at `~/miniconda3/envs/vision/bin/pdftotext` — NOT on PATH in a bare shell.
- No new dependencies. No pytest available — tests are standalone `python` scripts using bare `assert`.
- Not a git repository. Do NOT run `git` anything. Each task's "commit" step is replaced by a verification run.
- Hierarchy derives from CODE digits, never the dash count (handoff §3). Trust label ORDER, never the integer dash depth.
- Schema unchanged: `codes(pct_code, level, parent_code, description_raw, description_full, cd_percent, fiscal_year, is_synthetic)`. No new rows, no new columns.
- Do not edit `pct_corpus.db` by hand; rebuild via `build_codes.py`.
- The PDF source is `PTC-2025-26.pdf`. Use `build_codes.py --chapters 42,25,85,44,01,02` to limit rebuild scope while iterating.

---

### Task 1: Test harness + frozen fixtures of the current (buggy) behavior

Establishes the failing tests that encode the success criteria, plus reusable fixture text blocks lifted verbatim from the PDF. The fixtures are raw `pdftotext -layout` lines, so tests can parse them through `parse_with_labels` without the PDF.

**Files:**
- Create: `tests/fixtures.py`
- Create: `tests/test_dash_labels.py`

**Interfaces:**
- Produces: `tests/fixtures.py` exposing module-level strings `FIX_4202`, `FIX_2501`, `FIX_8544`, `FIX_4402`, `FIX_0205`, `FIX_0101` — each a multi-line raw-text block.
- Produces: helper `build_rows_from_text(raw: str) -> dict[str, dict]` in `tests/test_dash_labels.py` that calls `build_codes.parse_with_labels(raw)` then `build_codes.build_db(stream, chapter_titles={})` and returns the rows dict keyed by `pct_code`.

- [ ] **Step 1: Write the fixtures file**

Create `tests/fixtures.py`. Each block is copied verbatim from `pdftotext -layout PTC-2025-26.pdf` (column spacing preserved — it carries the indent the parser keys on for specific-duty detection, though label depth comes from dashes). Reproduce exactly:

```python
"""Raw pdftotext -layout line blocks lifted verbatim from PTC-2025-26.pdf.

Column spacing is intentional and must not be reflowed: the parser keys
specific-duty detection on absolute indent. Dash counts encode label nesting.
"""

FIX_4202 = """\
42.02       Trunks, suit- cases, vanity- cases, executive- cases,
            briefcases, school satchels, spectacle cases, binocular
            cases, camera cases, musical instrument cases, gun
            cases, holsters and similar containers; travelling- bags,
            insulated food or beverages bags, toilet bags,
            rucksacks, handbags, shopping- bags, wallets, purses,
            map- cases, cigarette- cases, tobacco- pouches, tool
            bags, sports bags, bottle- cases, jewellery boxes,
            powder- boxes, cutlery cases and similar containers, of
            leather or of composition leather, of sheeting of
            plastics, of textile materials, of vulcanised fibre or of
            paperboard, or wholly or mainly covered with such
            materials or with paper.

            - Trunks, suit- cases, vanity- cases, executive- cases, brief
            cases, school satchels and similar containers:
            - - With outer surface of leather or of composition leather:

4202.1120   - - - Suit-cases, of leather or composition leather             20
4202.1190   - - - Other                                                     20
            - - With outer surface of plastics or of textile materials:

4202.1210   - - - Travelling bags of plastics or textile materials          20
4202.1220   - - - Suit cases of plastics or textile materials               20
4202.1290   - - - Other                                                     20
4202.1900   - - Other                                                       20
            - Handbags, whether or not with shoulder strap, including
            those without handle:
4202.2100   - - With outer surface of leather or of composition leather     20

4202.2200   - - With outer surface of sheeting of plastics or of textile    20
            materials
4202.2900   - - Other                                                       20
            - Articles of a kind normally carried in the pocket or in the
            handbag:
4202.3100   - - With outer surface of leather or of composition leather     20

4202.3200   - - With outer surface of sheeting of plastics or of textile    20
            materials
4202.3900   - - Other                                                       20
            - Other:
4202.9100   - - With outer surface of leather or of composition leather     20

4202.9200   - - With outer surface of sheeting of plastics or of textile    20
            materials
4202.9900   - - Other                                                       20
"""

FIX_2501 = """\
25.01       Salt (including table salt and denatured salt) and pure
            sodium chloride, whether or not in aqueous solution or
            containing added anti- caking or free- flowing agents;
            sea water.
2501.0010   - - - Table salt                                           20
            - - - Rock salt:
2501.0021   - - - - Pink rock salt                                     20
2501.0029   - - - - Other                                              20
2501.0030   - - - Sea salt                                             20
2501.0090   - - - Other                                                20
"""

FIX_8544 = """\
85.44       Insulated wire, cable and other insulated electric
            conductors.
8544.2000   - Co- axial cable and other co- axial electric conductors        20

            - Ignition wiring sets and other wiring sets of a kind used in
            vehicles, aircraft or ships:
            - - - Of a kind used in vehicles of chapter 87:
8544.3011   - - - - Wiring sets and cable sets for vehicles of heading       35
            87.03 and vehicles of sub-headings 8704.2190, 8704.3130
8544.3019   - - - - Other                                                    35
8544.3090   - - - Other                                                      20
"""

FIX_4402 = """\
44.02        Wood charcoal (including shell or nut charcoal),
            whether or not agglomerated.
4402.1000    - Of bamboo                                                      0
4402.2000   - Of shell or nut                                                 0
4402.9000    - Other                                                          0
"""

FIX_0205 = """\
0205.0000   Meat of horses, asses, mules or hinnies, fresh, chilled    20
"""

FIX_0101 = """\
01.01       Live horses, asses, mules and hinnies.
            - Horses:
0101.2100   - - Pure-bred breeding animals                                0
0101.2900   - - Other                                                     0
0101.3000   - Asses                                                       0
"""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dash_labels.py`. These encode the spec's success criteria. Run via plain `python` (no pytest).

```python
#!/usr/bin/env python3
"""Standalone (no-pytest) tests for dash-label path reconstruction in build_codes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for build_codes
sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests dir for fixtures
import build_codes
import fixtures as F


def build_rows_from_text(raw: str) -> dict:
    stream = build_codes.parse_with_labels(raw)
    return build_codes.build_db(stream, chapter_titles={})


def _full(rows, code):
    return rows[code]["description_full"]


def test_4202_siblings_distinct():
    rows = build_rows_from_text(F.FIX_4202)
    a = _full(rows, "4202.2100")
    b = _full(rows, "4202.3100")
    c = _full(rows, "4202.9100")
    assert a != b != c and a != c, "4202.21/31/91 description_full must differ"
    assert "Handbags" in a, a
    assert "carried in the pocket" in b, b
    # 4202.91 sits under the bare "- Other:" group label
    assert b != c


def test_2501_rock_salt_in_chain():
    rows = build_rows_from_text(F.FIX_2501)
    assert "Rock salt" in _full(rows, "2501.0021"), _full(rows, "2501.0021")
    assert "Pink rock salt" in _full(rows, "2501.0021")
    # sibling NOT under rock salt must not gain it
    assert "Rock salt" not in _full(rows, "2501.0030"), _full(rows, "2501.0030")


def test_8544_both_labels_in_chain():
    rows = build_rows_from_text(F.FIX_8544)
    full = _full(rows, "8544.3011")
    assert "Ignition wiring sets" in full, full
    assert "Of a kind used in vehicles of chapter 87" in full, full


def test_4402_unchanged():
    rows = build_rows_from_text(F.FIX_4402)
    assert "Of bamboo" in _full(rows, "4402.1000")
    assert "Of shell or nut" in _full(rows, "4402.2000")
    assert "Other" in _full(rows, "4402.9000")
    # the three must differ
    s = {_full(rows, c) for c in ("4402.1000", "4402.2000", "4402.9000")}
    assert len(s) == 3, s


def test_0205_heading_is_leaf():
    rows = build_rows_from_text(F.FIX_0205)
    assert rows["0205.0000"]["level"] == "heading"
    assert "Meat of horses" in _full(rows, "0205.0000")


def test_0101_mixed_depths():
    rows = build_rows_from_text(F.FIX_0101)
    # 0101.21 under "- Horses:" group; 0101.30 "- Asses" is its own subheading
    assert "Horses" in _full(rows, "0101.2100"), _full(rows, "0101.2100")
    assert "Pure-bred breeding animals" in _full(rows, "0101.2100")
    assert "Asses" in _full(rows, "0101.3000"), _full(rows, "0101.3000")
    assert "Horses" not in _full(rows, "0101.3000"), _full(rows, "0101.3000")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests to confirm they fail against current code**

Run: `~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: `test_4402_unchanged`, `test_0205_heading_is_leaf` PASS; `test_4202_siblings_distinct`, `test_2501_rock_salt_in_chain`, `test_8544_both_labels_in_chain`, `test_0101_mixed_depths` FAIL (labels missing from chain). This proves the harness exercises the real bug.

- [ ] **Step 4: Verify (no commit — not a git repo)**

Run: `ls tests/` — confirm both files exist. Record which tests fail; that is the baseline.

---

### Task 2: Per-code label paths in `subheading_labels` (shared-prefix rule)

Replace the "deepest stack label with depth ≤ 2" heuristic with the shared-prefix-of-group rule, and return the full subheading label path (above-HS6) plus the per-leaf national label path (below-HS6).

**Files:**
- Modify: `build_codes.py` — rewrite `subheading_labels` (currently lines 268-302), add `_ordered_labels` helper.

**Interfaces:**
- Consumes: `leaf.stack` (dict `depth -> label`), `leaf.code`, `leaf.desc` from `parse_with_labels`.
- Produces: `subheading_labels(stream) -> dict[str, dict]` returning, per 6-digit subheading key `XXXX.YY`:
  - `"raw"`: the subheading's own `description_raw` (str)
  - `"path"`: above-HS6 ordered label list (list[str]) — the shared group labels ABOVE the subheading's own label
- Produces: `national_label_path(leaf, subheading_path) -> list[str]` returning the leaf's labels ordered by depth that are NOT part of `subheading_path` and are NOT the leaf's own inline label — i.e. group labels strictly between the subheading and the leaf.

- [ ] **Step 1: Write the failing unit test**

Append to `tests/test_dash_labels.py` (before `main`):

```python
def test_subheading_label_paths():
    stream = build_codes.parse_with_labels(F.FIX_4202)
    sl = build_codes.subheading_labels(stream)
    # 4202.21 is grouped under the depth-1 "Handbags..." label; its own label is
    # "With outer surface of leather..."
    assert "4202.21" in sl
    assert any("Handbags" in p for p in sl["4202.21"]["path"]), sl["4202.21"]
    assert "leather" in sl["4202.21"]["raw"].lower(), sl["4202.21"]
    # 4202.11 has TWO leaves (.1120/.1190) at depth-3 under "Trunks..." + a depth-2
    # split; the shared prefix is the depth-1 "Trunks..." label.
    assert any("Trunks" in p for p in sl["4202.11"]["path"]), sl["4202.11"]

def test_national_label_path_8544():
    stream = build_codes.parse_with_labels(F.FIX_8544)
    sl = build_codes.subheading_labels(stream)
    sub_path = sl["8544.30"]["path"]
    # find the 8544.3011 leaf object
    leaf = next(o for k, o in stream if k == "leaf" and o.code == "8544.3011")
    np = build_codes.national_label_path(leaf, sub_path + [sl["8544.30"]["raw"]])
    assert any("chapter 87" in x for x in np), np
```

- [ ] **Step 2: Run to confirm failure**

Run: `~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: `test_subheading_label_paths` and `test_national_label_path_8544` FAIL/ERROR (`subheading_labels` returns plain strings, no `national_label_path`).

- [ ] **Step 3: Implement the shared-prefix rule**

Replace the entire `subheading_labels` function (lines 268-302) with:

```python
def _ordered_labels(stack: dict[int, str]) -> list[str]:
    """Stack {depth: label} -> labels ordered shallow→deep. Depth VALUE is not
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
        # full ordered ancestry per leaf = stack labels + the leaf's own inline label
        ancestries = [
            _ordered_labels(lf.stack) + [extract_specific_duty(lf.desc, None)[0]]
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
            raw = first[len(prefix)]          # diverging label names the subheading
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
    full = _ordered_labels(leaf.stack) + [own]
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
```

- [ ] **Step 4: Run to confirm the two new unit tests pass**

Run: `~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: `test_subheading_label_paths` and `test_national_label_path_8544` PASS. The full-chain tests (4202/2501/8544/0101) may still FAIL — they depend on Task 3 wiring the paths into `description_full`. `test_4402_unchanged` and `test_0205_heading_is_leaf` still PASS.

- [ ] **Step 5: Verify (no commit)**

Re-read the diff of `build_codes.py`. Confirm only `subheading_labels` changed and two helpers were added; no other function touched.

---

### Task 3: Splice label paths into `description_full` and subheading `description_raw`

Wire the Task-2 paths through `build_db`: store them on rows, then rebuild `full()` to interleave subheading path + subheading raw + national path + national raw.

**Files:**
- Modify: `build_codes.py` — `build_db` (currently lines 305-398): consume the new `subheading_labels` shape, attach `_label_path` to national + subheading rows, rewrite `full()`.

**Interfaces:**
- Consumes: `subheading_labels(stream) -> {sub: {"raw","path"}}` and `national_label_path(leaf, chain)` from Task 2.
- Produces: rows with `description_full` containing the spliced label chain. Schema unchanged.

- [ ] **Step 1: Confirm the target full-chain tests currently fail**

Run: `~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: `test_4202_siblings_distinct`, `test_2501_rock_salt_in_chain`, `test_8544_both_labels_in_chain`, `test_0101_mixed_depths` FAIL.

- [ ] **Step 2: Update the leaf loop to capture national label paths and subheading raw**

In `build_db`, replace the subheading-label retrieval and the leaf loop (lines 320-339) with:

```python
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
```

- [ ] **Step 3: Update subheading synthesis to store its label path**

Replace the subheading synthesis block (lines 341-349) with:

```python
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
```

- [ ] **Step 4: Ensure heading rows carry an (empty) `_label_path`**

The heading loop (lines 309-317), chapter synthesis (351-358), and missing-heading synthesis (362-369) build row dicts WITHOUT `_label_path`. Add `_label_path=[]` to each of those three `dict(...)` constructions so `full()` can read the key uniformly. Example for the heading loop:

```python
        rows[code] = dict(
            pct_code=code, level="heading", parent_code=ch,
            description_raw=obj.desc, cd_percent=obj.cd, is_synthetic=0,
            _label_path=[],
        )
```

Apply the same `_label_path=[]` addition to the chapter-synthesis dict and the missing-4-digit-heading synthesis dict.

- [ ] **Step 5: Rewrite `full()` to splice label paths**

Replace `full()` (lines 372-392) with a version that, walking the parent chain, prepends each row's `_label_path` BEFORE that row's own `description_raw`:

```python
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
```

- [ ] **Step 6: Strip the internal `_label_path` key before writing rows**

`write_db` inserts named params; an extra dict key is harmless to `executemany` (it ignores unused keys), but to keep rows clean, after the `full()` loop (lines 394-396) add:

```python
    for code, r in rows.items():
        r["description_full"] = full(code)
        r["fiscal_year"] = FISCAL_YEAR
        r.pop("_label_path", None)
```

(Replace the existing two-line loop body with these three lines.)

- [ ] **Step 7: Run all tests**

Run: `~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: ALL tests PASS (`6/6` plus the 2 Task-2 unit tests = `8/8 passed`).

- [ ] **Step 8: Verify (no commit)**

Confirm test output reads `8/8 passed`. If any fail, debug the splice ordering before proceeding.

---

### Task 4: Full rebuild + corpus-level regression verification

Rebuild the real DB and verify the spec's corpus-level success criteria against the full PDF, not just fixtures.

**Files:**
- None modified. Runs `build_codes.py` and SQL checks.

**Interfaces:**
- Consumes: the finished `build_codes.py`.

- [ ] **Step 1: Rebuild the affected chapters first (fast iteration)**

Run:
```bash
~/miniconda3/envs/vision/bin/python build_codes.py --chapters 42,25,85,44,01,02
```
Expected: prints level counts and `orphan parent_code rows: 0`.

- [ ] **Step 2: Spot-check the headline defect is fixed**

Run:
```bash
~/miniconda3/envs/vision/bin/python -c "
import sqlite3; c=sqlite3.connect('pct_corpus.db')
for code in ('4202.2100','4202.3100','4202.9100','2501.0021','8544.3011'):
    print(code, '::', c.execute('SELECT description_full FROM codes WHERE pct_code=?',(code,)).fetchone()[0])
"
```
Expected: 4202.21/31/91 all DIFFERENT, containing Handbags / pocket / Other respectively; 2501.0021 contains "Rock salt"; 8544.3011 contains "Ignition wiring sets" AND "chapter 87".

- [ ] **Step 3: Full rebuild (all chapters)**

Run:
```bash
~/miniconda3/envs/vision/bin/python build_codes.py
```
Expected: `level counts:` chapter ~96, heading ~1228, subheading ~5403, national ~7380; `orphan parent_code rows: 0`. Counts within ±a handful of the documented baseline (schema/row-count unchanged is a success criterion).

- [ ] **Step 4: Measure duplicate-collapse improvement**

Run:
```bash
~/miniconda3/envs/vision/bin/python -c "
import sqlite3; c=sqlite3.connect('pct_corpus.db')
dup=c.execute('''SELECT COUNT(*) FROM (SELECT description_full FROM codes WHERE level=\"national\" GROUP BY description_full HAVING COUNT(*)>1)''').fetchone()[0]
tot=c.execute('SELECT COUNT(*) FROM codes WHERE level=\"national\"').fetchone()[0]
print(f'national rows: {tot}; description_full values shared by >1 national row: {dup}')
"
```
Expected: a small number of genuinely-duplicate groups (e.g. true "Other" leaves with no distinguishing ancestry). It should be FAR lower than before the fix. Eyeball the remaining dups:
```bash
~/miniconda3/envs/vision/bin/python -c "
import sqlite3; c=sqlite3.connect('pct_corpus.db')
for f,n in c.execute('SELECT description_full, COUNT(*) n FROM codes WHERE level=\"national\" GROUP BY description_full HAVING n>1 ORDER BY n DESC LIMIT 15'):
    print(n, '::', f[:120])
"
```
Confirm any remaining duplicates are genuinely identical lines in the PDF (e.g. distinct codes that legitimately share full text), not lost-label collisions.

- [ ] **Step 5: Re-run fixture tests against the final code**

Run: `~/miniconda3/envs/vision/bin/python tests/test_dash_labels.py`
Expected: `8/8 passed`.

- [ ] **Step 6: Verify (no commit)**

Summarize: level counts, orphan count (0), duplicate-full count before vs after. This is the completion evidence.

---

## Notes for the implementer

- The dash-stack reset on each new heading already happens in `parse_with_labels` (line 222). Do not change it.
- `extract_specific_duty` is applied to labels in Task 2 so Rs./MT smears don't leak into the path. Keep that call.
- If a fixture test fails because a label has slightly different whitespace than asserted, fix the ASSERTION substring, not the parser — the parser's whitespace collapse (`re.sub(r"\s+", " ", ...)`) is correct.
- `build_embeddings.py` is NOT run by this plan. After the DB is verified, embeddings must be regenerated separately (out of scope here).
