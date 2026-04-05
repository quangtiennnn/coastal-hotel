# Data Preparation Proposal: `data-prepare.ipynb`

## Goal

Merge `hotel.csv` (8,574 rows, 41 cols) and `distance2coast.csv` (29,446 rows, 6 cols) on `hotel_id` to produce a single enriched dataset.

---

## Key Observations

| File | Rows | Join Key | Notes |
|---|---|---|---|
| `hotel.csv` | 8,574 | `hotel_id` | Authoritative hotel metadata |
| `distance2coast.csv` | 29,446 | `hotel_id` | ~3.4× more rows → possible duplicate `hotel_id`s |

- `distance2coast.csv` duplicates `longitude` and `latitude` from `hotel.csv` — these will be dropped from the right side before merging.
- `distance2coast.csv` adds 3 new columns: `hotel_coordinate` (WKT POINT in WGS84), `distance2coastline` (meters), `nearest_coordinate` (WKT POINT in Web Mercator / EPSG:3857).

---

## Notebook Structure

### Cell 1 — Imports & Paths
```python
import pandas as pd
from pathlib import Path
DATA_DIR = Path("../data")
```

### Cell 2 — Load & Inspect
- Load both CSVs with `pd.read_csv`
- Print `.shape`, `.dtypes`, and `.head()` for each
- Check `hotel_id` uniqueness in each file

### Cell 3 — Deduplication Check (distance2coast)
- Count duplicate `hotel_id` values in `distance2coast.csv`
- If duplicates exist: inspect them and decide strategy (keep first / keep min distance / keep all)

### Cell 4 — Pre-merge Cleanup
- Drop duplicate columns `longitude` and `latitude` from `distance2coast` (keep from `hotel.csv`)

### Cell 5 — Merge
- **Join type**: `left` join (`hotel.csv` as left) on `hotel_id`
  - Rationale: preserve all 8,574 known hotels; hotels in `distance2coast` without a hotel record are not useful
- Result shape and null count check for distance columns

### Cell 6 — Validation
- Assert no duplicate `hotel_id` in output (after dedup step)
- Report: how many hotels have distance data vs. are missing it

### Cell 7 — Export
- Save merged dataframe to `data/hotel_with_distance.csv`
- Print final shape and sample rows

---

## Output

`data/hotel_with_distance.csv` — 8,574 rows, 44 columns:
- All 41 original `hotel.csv` columns
- 3 new columns from `distance2coast.csv`: `hotel_coordinate`, `distance2coastline`, `nearest_coordinate`

---

## Open Question

`distance2coast.csv` has 29,446 rows vs 8,574 hotels — the notebook will surface whether this is many-to-one (multiple distance records per hotel) and handle it explicitly rather than silently dropping rows.
