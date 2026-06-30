# PCT Corpus Build — Handoff Notes

Context: this is the first data-engineering step of an HS/PCT classification pipeline. Goal of this script: scrape the FBR Pakistan Customs Tariff PDF into two normalized tables (`codes`, `notes`). No models involved at this stage — pure parsing/data engineering.

---

## 1. PCT Code Structure (read this first)

PCT codes are 8 digits, formatted as `XXXX.XXXX`. The meaningful split is **2+2+2+2**, not 4+4:

| Digits | Level | Origin | Example (`0102.2110`) |
|---|---|---|---|
| 1–2 | Chapter | WCO international | `01` |
| 3–4 | Heading | WCO international | `0102` |
| 5–6 | Subheading (HS6) | WCO international | `0102.21` |
| 7–8 | National tariff line | Pakistan/FBR only | `0102.2110` |

The first **6** digits are identical to the international Harmonized System used by 200+ countries. Only the last 2 digits are Pakistan-specific (Pakistan adopted HS in 1988, contains ~5,200 international tariff lines extended to ~8,000+ at 8-digit level).

Why this matters for the corpus, not just trivia: WCO Explanatory Notes and GRI legal-interpretation text are written and published at the **6-digit subheading level**. The 7th–8th digit split is FBR's own enumeration (e.g. Bulls / Cows / Oxen / Other) and almost never has its own separate legal text — it inherits meaning from its subheading.

---

## 2. The HS6 problem: subheadings are not printed as their own rows

In the tariff PDF, only two row types carry a literal code:

- **4-digit bold heading rows** (e.g. `85.43`)
- **8-digit national-line rows** (e.g. `8543.7010`) — these always carry a CD% figure.

The 6-digit subheading (e.g. `0102.21`, `8543.70`) **never appears with its own printed code**. It only exists as an un-coded, dash-bulleted line sitting above the group of 8-digit rows that share that 6-digit prefix.

**Consequence: subheading rows must be derived, not scraped directly** — by grouping leaf (8-digit) rows by their shared first 6 digits and synthesizing one subheading record per group, pulling its description from the dash-label common to that group.

---

## 3. Dash-depth ≠ hierarchy depth (the part to be careful with)

It would be convenient if dash count mapped cleanly to hierarchy depth (1 dash = subheading digit 5, 2 dashes = digit 6, etc.), but it doesn't — confirmed against real rows:

- Some leaf codes sit **3 dashes deep with no 2-dash line above them at all** (heading `85.43`: `- Other machines and apparatus:` at 1 dash, then leaf codes appear directly at 3 dashes).
- Others go **4 dashes deep**, with an intermediate dash-line that has *no code of its own* (heading `85.44`: `- - - Of a kind used in vehicles of chapter 87:` sits between dash-1 and the dash-4 leaf codes `8544.3011`/`3012`/`3019`, purely as organizing text).

**The ground truth for hierarchy level is the numeric code itself (its digit count/prefix), not the dash count.** Dash-stack text should be tracked purely as a running stack (push/replace per depth, clear deeper levels on a new bullet) and used only to reconstruct full description text for whatever row comes next — including stacking through dash-levels that never get their own code.

The boundary between "this dash text belongs to the 6-digit subheading" vs. "this dash text is just 7th/8th-digit grouping context" has to be inferred by checking which leaf codes (by shared first-6-digit prefix) sit underneath it — not by counting dashes. You have the full PDF: inspect a range of chapters (84/85 — machinery/electrical — look like the deepest-nested ones) before locking in a single parsing rule.

---

## 4. Table schemas

### `codes`
```sql
pct_code          TEXT PRIMARY KEY   -- "01", "0102", "0102.21", "0102.2110"
level             TEXT               -- 'chapter' | 'heading' | 'subheading' | 'national'
parent_code       TEXT               -- FK to codes.pct_code, NULL for chapter
description_raw   TEXT               -- this row's own label only, dash-stripped
description_full  TEXT               -- denormalized: full ancestor chain joined, in order
cd_percent        TEXT               -- NULL except mostly at 'national', occasionally 'heading'
fiscal_year       TEXT               -- "2025-26"
is_synthetic      BOOLEAN            -- TRUE for derived subheading rows (see §2)
```

### `notes`
```sql
note_id      INTEGER PRIMARY KEY AUTOINCREMENT
ref_code     TEXT     -- FK to codes.pct_code (chapter level, mainly, for this scrape)
note_type    TEXT      -- 'section_note' | 'chapter_note' | 'explanatory_note'
text         TEXT
source       TEXT      -- "FBR Tariff 2025-26" | "WCO Explanatory Notes" | "conseric.pk" etc
fiscal_year  TEXT
```

---

## 5. Notes — sourcing and inheritance

- **Chapter notes**: not reliably present as clean, scrape-ready text inside this tariff PDF — may need sourcing from elsewhere (e.g. secondary per-chapter references like conseric.pk's blog posts, or other FBR-published chapter-note documents). Treat as a **separate follow-up ingestion task** — don't block the main 2-table scrape on this.
- **Section notes / WCO Explanatory Notes**: fully out of scope for this script — external sources, added in a later pass.
- **Inheritance, not duplication**: a note attaches once, at the level/code it legally applies to (`ref_code`). Anything downstream that needs "all notes relevant to this heading/national code" walks `parent_code` up the chain and collects every `notes` row whose `ref_code` appears in that ancestor chain. Never copy note text down into child rows.

---

## 6. Explicitly left to the code session's judgment

- Exact regex/parsing rules for dash-counting and code-prefix extraction — inspect several chapters (esp. 84/85) before fixing one rule.
- Where the subheading/national text boundary falls when an un-coded grouping dash-line appears mid-chain.
- Chapter-title sourcing (the PDF's chapter cover-page format, if usable) — flag back if inconsistent rather than guessing at it.
