# Data Description

## 1. Raw Inputs

| File | Rows | Description |
|---|---|---|
| `data/hotel.csv` | 8,574 | Full Agoda hotel catalogue for Vietnam (41 columns: name, star rating, room count, GPS, year opened/renovated, etc.) |
| `data/distance2coast.csv` | 29,446 | GIS-computed distance from each hotel GPS coordinate to the nearest coastline segment (km) |
| `data/agoda-reviews.csv` | 197,508 | Agoda reviews across all languages (25 columns: review text, score 0–10, stay detail string, reviewer nationality, travel group, etc.) |
| `scraping/outputs/hotel_*_reviews.json` | 466 files | Google Maps reviews scraped with Playwright, one JSON per hotel |

---

## 2. Hotel Metadata (`notebook 01`)

**Step:** Join `hotel.csv` + `distance2coast.csv` on `hotel_id`.

| Output | Hotels | Filter |
|---|---|---|
| `hotel_with_distance.csv` | 8,574 | All — adds `distance2coastline` (km), `hotel_coordinate`, `nearest_coordinate` |
| `hotel_filtered.csv` | **1,150** | `number_of_reviews > 500` — hotels with enough reviews for meaningful analysis |

All 8,574 hotels have a distance value (0 missing after join).

---

## 3. Agoda Reviews (`notebook 05`)

**Step:** Enrich raw Agoda reviews with hotel metadata and parse stay timing.

- Stay detail column (Vietnamese string: *"Đã ở 4 đêm vào Tháng 1 năm 2024"*) parsed into `stay_nights`, `stay_month`, `stay_year`, `stay_period` (YYYY-MM). 977 records (<0.5%) failed to parse and retain null temporal fields.
- Joined with `hotel_with_distance.csv` to add `distance2coastline` — 0 reviews missing distance after join.
- Score normalized from 0–10 → 1–5 scale in the merge step (notebook 07).

| Output | Rows | Description |
|---|---|---|
| `agoda-review-prepare.csv` | 197,508 | Full enriched dataset, all languages (32 columns) |
| `agoda-review-en-vi.csv` | 125,347 | English + Vietnamese only (86,322 en · 39,025 vi), trimmed to 22 columns for topic modeling |

**Language breakdown of raw Agoda corpus:**

| Language | Reviews |
|---|---|
| English | 86,322 |
| Japanese | 40,634 |
| Vietnamese | 39,025 |
| Chinese | 8,975 |
| Other / Unknown | 22,552 |
| **Total** | **197,508** |

**How many of the 197,508 Agoda reviews belong to the 1,150 analysis hotels?**

| Scope | Reviews |
|---|---|
| All 197,508 reviews (all hotels) | 197,508 |
| Reviews in 1,150 hotels (>500 reviews) | **113,629** |
| en + vi in 1,150 hotels | **69,129** |

---

## 4. Google Maps Reviews (`notebook 06`)

**Step:** Flatten 466 JSON files → structured DataFrame, enrich with hotel metadata.

- 184,962 raw rows extracted from 466 hotel JSON files (0 files failed to parse).
- Vietnamese relative timestamps (*"2 tháng trước"*) converted to approximate calendar dates using `dateutil.relativedelta` and the scrape timestamp; 46 records failed date parsing.
- Aspect sub-ratings extracted: room, service, location, food & drink, trip type, travel group (11 Vietnamese keys mapped to English columns).
- Joined with `hotel_with_distance.csv` — 0 reviews missing distance after join.
- Reviews with empty text body discarded (57,372 rating-only entries removed).

| Output | Rows | Description |
|---|---|---|
| `googlemaps-review-prepare.csv` | 184,962 | Full enriched dataset, all languages (43 columns) |
| `googlemaps-review-en-vi.csv` | 127,502 | English + Vietnamese with non-empty text (49,004 en · 78,498 vi), 31 columns |

**All 127,502 Google Maps en+vi reviews belong to the 1,150 analysis hotels** (the scraper was targeted at `hotel_filtered.csv`).

---

## 5. Merge & Preprocessing (`notebook 07`)

**Step:** Combine both sources → normalize → word-segment by language.

- Agoda score divided by 2 to convert from 0–10 → 1–5, aligning with Google Maps 1–5 rating.
- Both sources concatenated into one table; blank/null review texts dropped.
- `Preprocessor` class (`src/preprocessor.py`) applied:
  - **Vietnamese:** ViTokenizer (PyVi) word-segmentation — compound words joined with underscores (e.g., *khách_sạn*).
  - **English:** spaCy tokenization and normalization.
- Split by language into separate output files.

| Output | Rows | Description |
|---|---|---|
| `data-review-en-vi.csv` | 252,849 | Merged, before preprocessing |
| `final-reviews-en.csv` | 135,326 | Preprocessed English |
| `final-reviews-vi.csv` | 117,522 | Preprocessed Vietnamese |

### Important note on hotel scope (resolved downstream)

The merge step did **not** re-apply the 1,150-hotel filter. As a result, the CSV
exports include reviews from **1,704 unique hotels** (an extra ~1,090 small,
Agoda-only hotels leaked through). The reviews strictly within the 1,150 analysis
hotels are:

| | English | Vietnamese | Total |
|---|---|---|---|
| Within 1,150 hotels | 101,566 | 95,064 | **196,630** |
| All hotels in CSV exports | 135,326 | 117,522 | 252,848 |

The leak is corrected in the canonical DuckDB (`data/hotel_reviews.db`) by
re-applying the 1,150 filter, yielding **610 hotels** (only those with captured
en/vi reviews; 536 of the 1,150 have none). See `data-management/DATA_PROVENANCE.md`
for the full funnel.

---

## 6. DuckDB (`src/preprocess_to_duckdb.py`, `src/embed_to_duckdb.py`)

**Two db files exist** — use `data/hotel_reviews.db` as canonical:

| File | Size | State |
|---|---|---|
| `data/hotel_reviews.db` | ~1.9 GB | **canonical** — short reviews (<5 words) removed; full derived tables |
| `data-management/data/hotel_reviews.db` | ~118 MB | stripped snapshot — short reviews still present, no derived tables |

Reviews are encoded with `paraphrase-multilingual-mpnet-base-v2` (768-dim vectors).
Counts below are the **current** state of the canonical db — after short-review
removal **and** the destructive scope-fix that deleted the 50,735 leaked Agoda
rows (2026-06-22). The topic/label tables were wiped by that fix and must be
**regenerated** (refit notebook 21, then re-run silver labeling):

| Table | Rows | Description |
|---|---|---|
| `HOTEL` | 8,574 | Hotel metadata + distance to coast |
| `AGODA_REVIEW` | 62,129 | Agoda source rows (en + vi, 314 hotels) |
| `GOOGLEMAPS_REVIEW` | 112,242 | Google Maps source rows (en + vi, 417 hotels) |
| `REVIEW_DATA` *(view)* | 174,371 | Union of both sources — **610 hotels** |
| `REVIEW_TEXT_PROCESSED` | 174,371 | Normalized + word-segmented text |
| `REVIEW_EMBEDDINGS` | 174,371 | 768-dim float32 vectors |
| `REVIEW_TOPICS` | 0 | **wiped — regenerate via notebook 21** |
| `TOPIC_LABELS` | 0 | **wiped — regenerate via notebook 21** |
| `TOPIC_ASPECTS` | 0 | **wiped — re-run silver labeling** |

---

## 7. Final Corpus Summary

| Metric | Value |
|---|---|
| Hotels in Agoda catalogue | 8,574 |
| Hotels passing >500-review filter (`hotel_filtered.csv`) | **1,150** |
| Hotels selected for Google scrape | 503 |
| **Final analysis corpus — hotels** (canonical db, en+vi) | **610** |
| **Final analysis corpus — reviews** | **174,371** |
| — Agoda | 62,129 (314 hotels) |
| — Google Maps | 112,242 (417 hotels) |
| — On both platforms | 121 hotels |
| — English / Vietnamese | 89,163 / 85,208 |
| Hotels within 1 km of coast | 174 (28.5%), 45,949 reviews |
| Embeddings in canonical DuckDB | 225,106 |
| BERTopic models fitted | 20 (3 coast bands + 7 years × 2 languages) |
| Silver-labeled topics (`TOPIC_ASPECTS`) | 2,086 |
| LLM silver-label cost | ~$5.60 (Anthropic Batches API) |

> Pre-cleaning CSV exports report 252,848 reviews / 1,704 hotels (with the Agoda
> leak and short reviews still present). Those figures are superseded — see
> `data-management/DATA_PROVENANCE.md`.

---

## 8. Topic Modeling Scope (`notebook 11`)

20 BERTopic models fitted independently across two analysis dimensions:

| Dimension | Segments | Languages | Models |
|---|---|---|---|
| Distance to coastline | A (<0.1 km) · B (0.1–1.0 km) · C (≥1.0 km) | en, vi | 6 |
| Year | 2018 · 2019 · 2020 · 2021 · 2022 · 2023 · 2024 | en, vi | 14 |

---

## 9. Silver Labeling (`src/llm_label_submit.py`, `src/llm_label_retrieve.py`)

Discovered topics mapped to a 5-aspect taxonomy by Claude (claude-sonnet-4-6) via the Anthropic Message Batches API:

| Aspect | Description |
|---|---|
| `facility` | Physical infrastructure (rooms, pool, beach access) |
| `amenity` | Complimentary services (breakfast, wifi, parking) |
| `service` | Staff behavior and hospitality |
| `experience` | Atmosphere, views, overall guest experience |
| `loyalty` | Value for money, repeat-stay intent |

Each topic receives 1–3 aspects with per-aspect sentiment (positive / negative / neutral), weight, and optional sub-aspects. Results stored in DuckDB `TOPIC_ASPECTS` table.
