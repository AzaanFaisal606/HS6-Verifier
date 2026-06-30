# Design: build_codes.py dash-label path reconstruction

**Date:** 2026-06-29
**Status:** approved (design), pending implementation

## Problem

`description_full` (the text embedded into `vec_embeddings`) drops the
intermediate dash-label grouping lines that distinguish sibling subheadings.
Result: distinct tariff lines collapse to byte-identical embedding text.

### Confirmed defect (heading 42.02)

`4202.2100`, `4202.3100`, `4202.9100` all produce **identical** `description_full`:

```
Articles of leather... > Trunks, suit-cases... (full heading) > With outer surface of leather or of composition leather
```

The discriminating depth-1 group labels are lost:

- `4202.21` belongs under `- Handbags, whether or not with shoulder strap...:`
- `4202.31` belongs under `- Articles of a kind normally carried in the pocket...:`
- `4202.91` belongs under `- Other:`

These labels appear in the PDF as un-coded dash-bullet lines between the heading
and the leaves. The current parser captures them into a dash-stack but only ever
selects a *single* subheading label and discards the rest; `description_full`
walks the coded `parent_code` chain only (chapter → heading → subheading →
national), so the group labels — which are not rows — never enter the chain.

### Second defect class (heading 25.01, national-level groups)

`25.01` has a single HS6 subheading `2501.00`; the national lines carry their own
dash sub-grouping:

```
2501.0010   - - - Table salt
            - - - Rock salt:          <- national-level group label (depth 3)
2501.0021   - - - - Pink rock salt    <- depth 4
2501.0029   - - - - Other             <- depth 4
2501.0030   - - - Sea salt
```

The "Rock salt:" label (below HS6, grouping 7th/8th digits) is captured by the
stack but never injected into `2501.0021`/`2501.0029`'s `description_full`.

### Dash-depth is unreliable (heading 85.44)

```
            - Ignition wiring sets ... vehicles, aircraft or ships:   <- depth 1  (== HS6 8544.30)
            - - - Of a kind used in vehicles of chapter 87:           <- depth 3  (national group; depth-2 SKIPPED)
8544.3011   - - - - Wiring sets ...                                   <- depth 4
8544.3090   - - - Other                                              <- depth 3
```

Depths are non-contiguous (1 → 3, no depth-2 label). Any rule keyed on the
numeric depth value breaks here. **Only depth ORDERING is trustworthy**
(deeper = more specific), never the depth NUMBER. This matches handoff §3:
hierarchy comes from code digits, never the dash count.

## Decisions (from brainstorming)

1. **Representation:** inject the missing labels into `description_full` only.
   No schema change, no new rows, no new columns. Subheading `description_raw`
   also gets its parent group-label context where it would otherwise be a
   generic word like "Other".
2. **Scope:** reconstruct the FULL dash-label path — both above-HS6
   (subheading-grouping) and below-HS6 (national-grouping) labels.

## Architecture

### 1. Keep the dash-stack primitive

`parse_with_labels` already snapshots the running dash-stack
(`depth -> label`) onto each leaf as `leaf.stack`, and resets it on each new
heading. This is correct and stays. Sorted by depth, a leaf's stack is its
ordered label ancestry, independent of depth gaps:

- `4202.2100` stack: `{1: "Handbags...:", 2: "With outer surface of leather..."}`
- `8544.3011` stack: `{1: "Ignition wiring sets...:", 3: "Of a kind used in vehicles of ch87:"}`
- `2501.0021` stack: `{3: "Rock salt:"}` (plus whatever is above)

We trust the **order** of `sorted(stack)`, not the integer depths.

### 2. Split each leaf's label path at the HS6 boundary using code digits

Per HS6 group (all leaves sharing `XXXX.YY`, digits 1–6):

- **Subheading label path** = the ordered labels that are common to ALL leaves
  in the group (the shared prefix of their sorted label stacks). Labels above
  the digit-5/6 split are necessarily shared by every leaf in the subheading;
  labels below are not. This is the robust test — no depth arithmetic.
  - 4202.21 (single depth-2 leaf): shared prefix = `["Handbags...:"]`; the
    leaf's own depth-2 label becomes the subheading `description_raw`.
  - 8544.30: shared prefix = `["Ignition wiring sets...:"]`.
  - 4402.10 (single depth-1 leaf, no group label): shared prefix = `[]`;
    subheading uses the leaf's own label "Of bamboo" (unchanged behavior).
- **National label path** = the labels in a leaf's stack that are DEEPER (later
  in sort order) than the subheading path — the group-specific divergence.
  - `8544.3011`: `["Of a kind used in vehicles of ch87:"]`.
  - `2501.0021`: `["Rock salt:"]`.

### 3. Rebuild description_full to splice the label paths in

`description_full` is still the ancestor chain, but now interleaves the
reconstructed label paths at the correct positions:

```
chapter > heading > [subheading label path...] > subheading_raw > [national label path...] > national_raw
```

For `4202.2100`:
```
Articles of leather... > Trunks, suit-cases... > Handbags...: > With outer surface of leather or of composition leather
```
For `2501.0021`:
```
Salt... > Salt (heading) > Salt (subheading 2501.00) > Rock salt > Pink rock salt
```

Adjacent-duplicate collapse (already present) still applies, so a single-leaf
subheading whose label equals its leaf does not double-print.

### 4. Edge cases preserved (no regression)

- `.0000` heading-is-leaf (e.g. `0205.0000 Meat of horses...`, flush, no dash):
  no label path, treated as the heading row. Unchanged.
- Single-dash leaf directly under heading with no group label (4402): empty
  shared prefix, subheading = leaf's own label. Unchanged.
- Specific-duty Rs./MT smears (ch 15 oils): the existing
  `extract_specific_duty` strip still runs on every label and description before
  it enters a chain.
- Heading title overrides and synthesized headings: unchanged.

## Components touched

| Unit | Change |
|---|---|
| `parse_with_labels` | unchanged (stack snapshot already correct) — verify reset-on-heading still holds |
| `subheading_labels` | replace "deepest label ≤ depth 2" heuristic with shared-prefix-of-group rule; return BOTH the subheading `description_raw` and the subheading label path |
| new helper `national_label_path(leaf, subheading_path)` | labels deeper than the subheading prefix |
| `build_db` | store per-code label paths; rewrite `full()` to splice subheading + national label paths into the chain |
| `description_full` (`full()`) | interleave label paths at correct positions; keep adjacent-dup collapse |

## Success criteria

1. `4202.2100`, `4202.3100`, `4202.9100` have **distinct** `description_full`,
   each containing its group label (Handbags / pocket articles / Other).
2. `2501.0021` `description_full` contains "Rock salt".
3. `8544.3011` `description_full` contains both "Ignition wiring sets" and
   "Of a kind used in vehicles of chapter 87".
4. No two `national` rows share an identical `description_full` unless they are
   genuinely identical in the PDF (spot-check: count of duplicate
   `description_full` among `national` rows drops sharply vs current build).
5. Level counts stay in the same ballpark (96 chapter / ~1228 heading /
   ~5403 subheading / ~7380 national); orphan parent_code rows = 0.
6. Regression fixtures (4402, 0205.0000, 0101.30) unchanged.

## Out of scope

- Schema changes, new rows/columns.
- Re-running `build_embeddings.py` (separate step after build verified).
- Fixing the garbled ch-15 oil `cd_percent` smears (tracked in docs/issues.md).
