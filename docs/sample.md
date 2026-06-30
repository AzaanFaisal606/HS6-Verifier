# Sample `codes` Rows

Representative entries from `pct/pct_corpus.db` (table `codes`), chosen to show every
`level`, the parent chain, and the tricky parsing cases. Full untruncated values.

## Full ancestor chain — one national line walked up to its chapter (Live bovine animals → Bulls)

### `01` — chapter

| field | value |
|---|---|
| pct_code | 01 |
| level | chapter |
| parent_code | *(null)* |
| description_raw | Live animals |
| description_full | Live animals |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 1 |

### `01.02` — heading

| field | value |
|---|---|
| pct_code | 01.02 |
| level | heading |
| parent_code | 01 |
| description_raw | Live bovine animals. |
| description_full | Live animals > Live bovine animals. |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

### `0102.21` — subheading

| field | value |
|---|---|
| pct_code | 0102.21 |
| level | subheading |
| parent_code | 01.02 |
| description_raw | Pure-bred breeding animals: |
| description_full | Live animals > Live bovine animals. > Pure-bred breeding animals: |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 1 |

### `0102.2110` — national

| field | value |
|---|---|
| pct_code | 0102.2110 |
| level | national |
| parent_code | 0102.21 |
| description_raw | Bulls |
| description_full | Live animals > Live bovine animals. > Pure-bred breeding animals: > Bulls |
| cd_percent | 0 |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

## Deep-nest case A (handoff §3): leaf at 3 dashes with NO 2-dash line above. Subheading 8543.70 derived from the 1-dash grouping label.

### `85.43` — heading

| field | value |
|---|---|
| pct_code | 85.43 |
| level | heading |
| parent_code | 85 |
| description_raw | Electrical machines and apparatus, having individual functions, not specified or included elsewhere in this Chapter. |
| description_full | Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles > Electrical machines and apparatus, having individual functions, not specified or included elsewhere in this Chapter. |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

### `8543.70` — subheading

| field | value |
|---|---|
| pct_code | 8543.70 |
| level | subheading |
| parent_code | 85.43 |
| description_raw | Other machines and apparatus: |
| description_full | Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles > Electrical machines and apparatus, having individual functions, not specified or included elsewhere in this Chapter. > Other machines and apparatus: |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 1 |

### `8543.7010` — national

| field | value |
|---|---|
| pct_code | 8543.7010 |
| level | national |
| parent_code | 8543.70 |
| description_raw | Remote control |
| description_full | Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles > Electrical machines and apparatus, having individual functions, not specified or included elsewhere in this Chapter. > Other machines and apparatus: > Remote control |
| cd_percent | 5 |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

## Deep-nest case B (handoff §3): an un-coded 3-dash grouping line ('Of a kind used in vehicles of chapter 87:') hangs above the 4-dash leaves but must be IGNORED — the HS6 subheading 8544.30 takes the depth-1 label instead.

### `85.44` — heading

| field | value |
|---|---|
| pct_code | 85.44 |
| level | heading |
| parent_code | 85 |
| description_raw | Insulated (including enamelled or anodised) wire, cable (including co- axial cable) and other insulated electric conductors, whether or not fitted with connectors; optical fibre cables, made up of individually sheathed fibres, whether or not assembled with electric conductors or fitted with connectors. |
| description_full | Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles > Insulated (including enamelled or anodised) wire, cable (including co- axial cable) and other insulated electric conductors, whether or not fitted with connectors; optical fibre cables, made up of individually sheathed fibres, whether or not assembled with electric conductors or fitted with connectors. |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

### `8544.30` — subheading

| field | value |
|---|---|
| pct_code | 8544.30 |
| level | subheading |
| parent_code | 85.44 |
| description_raw | Ignition wiring sets and other wiring sets of a kind used in vehicles, aircraft or ships: |
| description_full | Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles > Insulated (including enamelled or anodised) wire, cable (including co- axial cable) and other insulated electric conductors, whether or not fitted with connectors; optical fibre cables, made up of individually sheathed fibres, whether or not assembled with electric conductors or fitted with connectors. > Ignition wiring sets and other wiring sets of a kind used in vehicles, aircraft or ships: |
| cd_percent | *(null)* |
| fiscal_year | 2025-26 |
| is_synthetic | 1 |

### `8544.3011` — national

| field | value |
|---|---|
| pct_code | 8544.3011 |
| level | national |
| parent_code | 8544.30 |
| description_raw | Wiring sets and cable sets for vehicles of heading 87.03 and vehicles of sub-headings 8704.2190, 8704.3130, 8704.3150, 8704.3190 and vehicles of heading 87.11 |
| description_full | Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles > Insulated (including enamelled or anodised) wire, cable (including co- axial cable) and other insulated electric conductors, whether or not fitted with connectors; optical fibre cables, made up of individually sheathed fibres, whether or not assembled with electric conductors or fitted with connectors. > Ignition wiring sets and other wiring sets of a kind used in vehicles, aircraft or ships: > Wiring sets and cable sets for vehicles of heading 87.03 and vehicles of sub-headings 8704.2190, 8704.3130, 8704.3150, 8704.3190 and vehicles of heading 87.11 |
| cd_percent | 35 |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

## Heading-is-leaf: a 4-digit heading printed directly as an 8-digit `.0000` line, carrying its own CD. Parent is the chapter (no subheading).

### `0205.0000` — heading

| field | value |
|---|---|
| pct_code | 0205.0000 |
| level | heading |
| parent_code | 02 |
| description_raw | Meat of horses, asses, mules or hinnies, fresh, chilled or frozen. |
| description_full | Meat and edible meat offal > Meat of horses, asses, mules or hinnies, fresh, chilled or frozen. |
| cd_percent | 20 |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

## Specific-duty: CD is a Rupee-per-tonne rate, not a percent. Stored as text in cd_percent; description stays clean.

### `1507.1000` — national

| field | value |
|---|---|
| pct_code | 1507.1000 |
| level | national |
| parent_code | 1507.10 |
| description_raw | Crude oil, whether or not degummed |
| description_full | Animal or vegetable fats and oils and their cleavage products; prepared edible fats; animal or vegetable waxes > Soya- bean oil and its fractions, whether or not refined, but not chemically modified. > Crude oil, whether or not degummed |
| cd_percent | Rs.10550/MT |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |

## Long real-world description (ch 87 cars) — kept intact, code-references inside the text preserved.

### `8703.2113` — national

| field | value |
|---|---|
| pct_code | 8703.2113 |
| level | national |
| parent_code | 8703.21 |
| description_raw | Mini Vans (CBU) |
| description_full | Vehicles other than railway or tramway rolling-stock, and parts and accessories thereof > Motor cars and other motor vehicles principally designed for the transport of persons (other than those of heading 87.02), including station wagons and racing cars. > Of a cylinder capacity not exceeding 1,000cc: > Mini Vans (CBU) |
| cd_percent | 50 |
| fiscal_year | 2025-26 |
| is_synthetic | 0 |
