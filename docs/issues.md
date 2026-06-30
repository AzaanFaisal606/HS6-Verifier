# Known Issues — `codes` Table Build + Retrieval

Issues 1–2 are `codes`-table data-quality limits. Issues 3–5 are
embedding/retrieval limits found while building `test_infer.py` /
`test_infer_dense.py`. Issue 2 is RESOLVED; the rest are open.

---

## Issue 1 — Mangled rupee-rate (`cd_percent`) on 6 ch-15 oil rows

A handful of national lines carry a **specific duty** (Rupees per metric tonne)
instead of an ad-valorem percent. `pdftotext -layout` splits and interleaves the
rate fragments across several far-right lines, so the reconstructed `cd_percent`
string is garbled. The `description_raw` for these rows is **clean** — only
`cd_percent` is affected.

| pct_code | description_raw | cd_percent (garbled) | likely intended |
|---|---|---|---|
| 1511.9010 | Palm stearin | `Rs.1 Rs.9 Rs.1 Rs.9 0800 050/ 0800 050/ /MT MT / MT MT` | ~Rs.10800 / Rs.9050 per MT |
| 1512.1100 | Crude oil | `Rs.1 Rs.1 6800 5000 /MT /MT` | ~Rs.16800 / Rs.15000 per MT |
| 1512.2100 | Crude oil, whether or not gossypol has been removed | `Rs.15000/M T` | Rs.15000 / MT |
| 1515.1100 | Crude oil | `Rs.1 Rs.9 0800 500/ /MT MT` | ~Rs.10800 / Rs.9500 per MT |
| 1515.2100 | Crude oil | `Rs.1 Rs.1 6600 5000 /MT /MT` | ~Rs.16600 / Rs.15000 per MT |
| 1515.3000 | Castor oil and its fractions | `Rs.9050/ Rs.905 Rs.905 MT 0/MT 0/MT` | ~Rs.9050 per MT |

Cleanly-reconstructed specific-duty rows (for reference, NOT broken):
`1507.1000 = Rs.10550/MT`, `1507.9000 = Rs.11700/MT`, `1512.2900 = Rs.16800/MT`,
`3706.1000 / 3706.9000 = Rs. 5 per meter`, `8517.13xx / 8517.14xx = Rs.250/set`.

**Why it happens:** these rows often list two alternating rates (e.g. a standard
rate and a concessionary rate) stacked vertically in one cell; pdftotext flattens
the two columns of digits into a single interleaved stream.

**Fix options (later pass):**
- Re-extract these specific pages with pdfplumber using word x/y coordinates to
  recover the true two-column rate layout.
- Or hand-correct the 6 values against the published FBR tariff PDF.

**Detection query:**
```sql
SELECT pct_code, description_raw, cd_percent
FROM codes
WHERE cd_percent LIKE 'Rs%' AND cd_percent LIKE '% %';
```

---

## Issue 2 — Heading `11.03` has no printed title — RESOLVED

Heading `11.03` is **never printed as its own `XX.XX` row** in the tariff body;
it appears only via its leaf rows (`1103.1100`, `1103.1300`, `1103.1900`,
`1103.2000`). The build synthesizes the heading row to keep the parent chain
intact, but the PDF gives it no title text.

**Resolution:** backfilled with the WCO title via `HEADING_TITLE_OVERRIDES` in
`build_codes.py`. The synthesizer now stamps any title-less synthetic heading
from that map.

| pct_code | level | parent_code | description_raw | is_synthetic |
|---|---|---|---|---|
| 11.03 | heading | 11 | Cereal groats, meal and pellets. | 1 |

There are now **0** title-less rows in the table. To add more such overrides
later, extend the `HEADING_TITLE_OVERRIDES` dict.

**Detection query (should return 0 rows):**
```sql
SELECT pct_code, level, parent_code
FROM codes
WHERE level = 'heading' AND (description_raw = '' OR description_raw IS NULL);
```

---

## Issue 3 — ~987 subheading rows have empty `description_raw` — OPEN (by design)

When a 6-digit subheading's national leaves diverge into different branches with
no single shared label (e.g. `8544.30`: leaves under "Of a kind used in vehicles
of ch87:" AND under "Other"), `subheading_labels` in `build_codes.py` sets the
subheading's `description_raw = ""` — there is no one label that names the whole
subheading. ~987 of 5403 subheadings are affected.

**Not a bug:** the discriminating grouping labels are NOT lost — they remain in
the `description_full` chain of the subheading's national children (and of the
subheading row itself). The empty value only means "this HS6 group has no single
own title in the PDF." `description_full` is never empty for these.

**Impact:** the `test_infer_dense.py` cascade deliberately does NOT query
subheading rows (subheading is not a stage), so empty subheading raws don't hurt
retrieval. Any future code that keys on subheading `description_raw` must handle
the empty case (fall back to `description_full` or the children's labels).

**Detection query:**
```sql
SELECT COUNT(*) FROM codes WHERE level='subheading'
AND (description_raw='' OR description_raw IS NULL);   -- ~987
```

---

## Issue 4 — Retrieval depends on which JSON field holds the discriminator — OPEN

The VLM emits both `function` (usage, e.g. "carried in a pocket") and
`embedding_description` (appearance). Only `embedding_description` is embedded for
retrieval. The Thinking model **inconsistently** places the functional
discriminator: sometimes in `embedding_description` (then `4202.3100` for a wallet
ranks #1), sometimes only in `function` (then it falls out of top-5). At
`temperature=0.3` the same image gives different ranks across runs.

**Root cause:** field-placement nondeterminism, NOT a corpus or parser defect.
The corpus text is correct (`4202.3100` = "Articles … carried in the pocket: >
With outer surface of leather"); the query sometimes just lacks the matching
phrase.

**Candidate fixes (not yet applied):**
- Embed `function` + `embedding_description` concatenated (one-line change in the
  client `main()`), so the functional signal is always in the query.
- Lower `temperature` toward 0 for more deterministic field placement.
- Prompt engineering to force usage/storage wording into `embedding_description`
  (tried partially in `test_infer.py` PROMPT — only partly effective).

---

## Issue 5 — Cascade heading stage can mis-rank; relies on beam width — OPEN (mitigated)

In `test_infer_dense.py`, stage 2 scores each heading on its OWN description. A
short generic heading (`42.05 "Other articles of leather or of composition
leather"`) can out-score the correct long enumerated heading (`42.02 "Trunks,
suit-cases … wallets, purses …"`) because the right keyword is diluted in 90
words. If the correct heading is pruned at stage 2, stage 3 can never recover it
— the cascade's structural weakness vs flat KNN.

**Mitigation in place:** widened beam — `N_HEAD=3` (and `N_CHAP=2`) keeps the
correct heading in the candidate set even when not ranked #1; stage 3 then KNNs
nationals across ALL kept headings and the strong national match wins. Verified:
wallet → `4202.3100` #1 with `N_HEAD=3` (was pruned at `N_HEAD=1`).

**Residual risk:** an image whose correct heading ranks below `N_HEAD` is still
lost. Raise the beam knobs if a known-correct code is missing from stage-3
output. A more robust (unbuilt) alternative: score each heading by its
best-matching national child (max-pooling) instead of its own diluted text.
