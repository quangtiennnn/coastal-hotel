# Coastal Hotel Review Analysis Pipeline

End-to-end pipeline for scraping, preprocessing, and analyzing Google Maps & Agoda reviews of Vietnamese coastal hotels. Combines async Playwright scraping, multilingual NLP, BERTopic topic modeling, DuckDB analytics, and Claude API silver-labeling.

**Stack:** Python 3.14 · uv · Playwright · BERTopic · sentence-transformers · spaCy + pyvi · DuckDB · Anthropic Claude API

---

## Setup

```bash
uv sync
uv run playwright install chrome
cp .env.example .env   # add ANTHROPIC_API_KEY
```

---

## Pipeline Overview

```text
Raw Scraping → Data Prep → DB Population → Topic Modeling → Analysis & Visualization
  (Scripts)    (NB 01–07)   (src/ scripts)   (NB 08–14)       (NB 12, 15–18)
```

---

## Step 1 — Raw Data Acquisition

**Scripts:** `scraping/get_reviews.py`, `scraping/run.py`, `scraping/get_metadata.py`

Scrape all hotel reviews from Google Maps and collect hotel metadata.

```bash
# Batch-scrape reviews for all hotels in scraping/hotels_processed.csv
uv run python scraping/run.py

# Scrape a single contributor profile (debugging / visible browser)
uv run python scraping/get_reviews.py "<URL>" --no-headless --output output.json

# Scrape hotel metadata (name, rating, phone, facilities, price)
uv run python scraping/get_metadata.py   # reads data/hotel_filtered.csv
```

**Outputs:**

- `scraping/outputs/hotel_{id}_reviews.json` — per-hotel review JSON (466 files)
- `scraping/outputs/all_hotels_reviews.json` — merged summary

**Scraper behavior:**

- Persistent Chrome session with anti-detection flags
- Scrolls each review panel for 60 seconds to trigger lazy-loading
- Expands all "See More" buttons before parsing HTML
- Skips hotels with an existing output file (idempotent)
- `SorryPage` (Google rate-limit) → skips hotel and continues

**Pre-scraped external inputs (not re-scraped):**

- `data/hotel.csv` — 8,574 Agoda hotels
- `data/distance2coast.csv` — GIS-computed coastal distances (29,446 rows)
- `data/agoda-reviews.csv` — 197,508 Agoda reviews (all languages)

---

## Step 2 — Data Preparation

**Notebooks:** `01` → `02` → `03` → `05` → `06` → `07`

### 01 — Merge hotels with distance data

Joins `hotel.csv` (8,574 rows) with `distance2coast.csv` on `hotel_id`.

| Output | Rows | Filter |
|---|---|---|
| `data/hotel_with_distance.csv` | 8,574 | All hotels |
| `data/hotel_filtered.csv` | 1,150 | >500 reviews AND ≤0.5 km from coastline |

### 02 — Flatten Google Maps JSONs

Parses the 466 raw JSON files from Step 1 into a flat CSV, merged with hotel metadata.

**Output:** `data/reviews_merged.csv` (184,962 rows)  
Columns include: `review_id`, `rating`, `review_text`, `aspect_rating` (room / service / location / food / atmosphere), `hotel_respond`

### 03 — Parse Agoda stay timing

Extracts stay metadata from Vietnamese Agoda strings (`"Đã ở X đêm vào Tháng M năm YYYY"`) into structured columns: `stay_nights`, `stay_month`, `stay_year`, `stay_period` (YYYY-MM format).

**Output:** `data/agoda-reviews-en-vi-parsed.csv` (125,347 rows)

### 05 — Prepare Agoda reviews

Enriches and filters raw Agoda reviews:
- Joins with hotel distance metadata
- Filters to English + Vietnamese only (86,322 en · 39,025 vi)
- Trims columns for topic modeling

| Output | Rows × Cols | Description |
|---|---|---|
| `agoda-review-prepare.csv` | 197,508 × 32 | Full enriched dataset |
| `agoda-review-en-vi.csv` | 125,347 × 22 | Topic-ready en/vi only |

### 06 — Prepare Google Maps reviews

Enriches and filters scraped Google Maps reviews:
- Parses Vietnamese relative timestamps → `approx_review_date`, `review_year`, `review_month`
- Extracts aspect ratings (11 Vietnamese keys → English columns)
- Joins with hotel metadata (distance, star_rating, accommodation_type)
- Filters to en/vi with non-empty review text

| Output | Rows × Cols | Description |
|---|---|---|
| `googlemaps-review-prepare.csv` | 184,962 × 43 | Full enriched dataset |
| `googlemaps-review-en-vi.csv` | 127,502 × 31 | Topic-ready en/vi only |

### 07 — Merge & preprocess

Combines both sources, normalizes rating scales (Agoda 0–10 → 1–5), then applies the `Preprocessor` class (`src/preprocessor.py`):
- **Vietnamese:** ViTokenizer word-segmentation (compounds joined with underscores: `khách_sạn`)
- **English:** spaCy tokenization

| Output | Rows | Description |
|---|---|---|
| `dataset-prepare/data-review-en-vi.csv` | 252,849 | Merged, before preprocessing |
| `dataset-prepare/final-reviews-en.csv` | 135,326 | Preprocessed English |
| `dataset-prepare/final-reviews-vi.csv` | 116,756 | Preprocessed Vietnamese |

---

## Step 3 — Database Population

**Scripts:** `src/preprocess_to_duckdb.py`, `src/embed_to_duckdb.py`

Loads all preprocessed reviews into DuckDB (`data/hotel_reviews.db`) and computes 768-dim sentence embeddings.

```bash
# Normalize + word-segment all 251,328 reviews (~6 min)
uv run python src/preprocess_to_duckdb.py

# Encode texts with paraphrase-multilingual-mpnet-base-v2 (~38 min on CPU)
uv run python src/embed_to_duckdb.py
```

Both scripts are **idempotent** — they skip rows already processed, safe to resume.

**DuckDB schema after this step:**

| Table | Rows | Description |
|---|---|---|
| `HOTEL` | 8,574 | Hotel metadata + distance to coast |
| `AGODA_REVIEW` | 125,347 | Agoda reviews (en + vi) |
| `GOOGLEMAPS_REVIEW` | 125,981 | Google Maps reviews (en + vi) |
| `REVIEW_DATA` *(view)* | 251,328 | Union of both sources |
| `REVIEW_TEXT_PROCESSED` | 251,328 | Normalized + word-segmented text |
| `REVIEW_EMBEDDINGS` | 251,328 | 768-dim float32 vectors |

---

## Step 4 — Topic Modeling

**Notebooks:** `08` → `09` → `10` → `11` → `14`

### 08 — Topic preparation

Loads reviews and embeddings from DuckDB into memory for downstream BERTopic runs.

### 09 — BERTopic demo

Fits BERTopic on a filtered subset using the full pipeline:
1. Encode `processed_text` with `paraphrase-multilingual-mpnet-base-v2` (768-dim)
2. Reduce with UMAP (`n_neighbors=15`, `n_components=5`, cosine metric)
3. Cluster with HDBSCAN (`min_cluster_size=50` en / `20` vi, EOM method)
4. Represent topics with c-TF-IDF + KeyBERTInspired + MaximalMarginalRelevance (`diversity=0.3`)

### 10 — Full preprocessing & embedding cache

Processes the full corpus (124,603 reviews), caches embeddings to `.npy`, and experiments with filtered subsets (province, star rating, region, distance band).

### 11 — Production topic runs (coast bands + years)

Fits **20 BERTopic models** independently across two dimensions:

| Dimension | Segments | Languages | Models |
|---|---|---|---|
| Distance to coast | A (<0.1 km) · B (0.1–0.5 km) · C (≥0.5 km) | en + vi | 6 |
| Year | 2018 · 2019 · 2020 · 2021 · 2022 · 2023 · 2024 | en + vi | 14 |

**Outputs:**
- `checkpoints/coast_band_{A,B,C}_{en,vi}.pkl` — model checkpoints (200 MB – 1.5 GB each)
- `checkpoints/year_{2018..2024}_{en,vi}.pkl` — yearly checkpoints
- DuckDB `TOPIC_LABELS` — one row per (run_id, topic_id) with top-100 words, seed_topic, seed_score
- DuckDB `REVIEW_TOPICS` — per-review topic assignments and probabilities

### 14 — Quick 10k diagnostic

Fits BERTopic on a random 10,000-sample per language (`seed=42`). Useful for fast hyperparameter testing before committing to full production runs.

---

## Step 5 — Analysis & Visualization

**Notebooks:** `12`, `15`, `16`, `17`, `18`

### 12 — SQL analysis

Interactive DuckDB queries exploring:
- Hotel distribution by city, star rating, distance band
- Review volume, language split, source breakdown per year
- Rating distributions (Agoda vs Google Maps)
- Aspect tag fill rates (room / service / location / food)
- Top 10 hotels by review count and average rating

### 15 — Coast word cloud banners

Generates publication-ready word cloud banners from the 3 distance-band BERTopic models.

```
Band C (50% width)   Band B (30%)    Band A (20%)
inland ≥0.5 km       0.1–0.5 km      beachfront <0.1 km
colormap: afmhot     gist_earth       PuBuGn
🌆 inland ←────────────────────────→ 🏖 beachfront
```

Each band: top 10 words from top 10 topics → rank-weighted frequency → WordCloud (300 max words).

**Outputs:** `IMG/coast_wordcloud_en.png`, `IMG/coast_wordcloud_vi.png` (26" × 7" @ 150 dpi)

### 16 — Database audit

Full schema inspection and data quality check across all 8 DuckDB tables:

- Row counts, DESCRIBE output, sample rows per table
- REVIEW_TOPICS: topic counts, outlier rates, run_id inventory
- REVIEW_DATA: year distribution (2018–2024), distance band split, source/language balance
- Silver-label readiness check (presence of `key_aspect`, `sentiment`, `confidence` columns)

### 17 — Silver label generation (Claude API)

Submits topics to Claude claude-sonnet-4-6 via the Anthropic Batches API for taxonomy labeling (~50% cheaper than standard API).

**Taxonomy — 5 key aspects:**

| Aspect | Description |
|---|---|
| `facility` | Physical infrastructure (rooms, pool, beach access) |
| `amenity` | Complimentary services (breakfast, wifi, parking) |
| `service` | Staff behavior and hospitality |
| `experience` | Atmosphere, views, overall guest experience |
| `loyalty` | Value for money, repeat-stay intent |

Each topic receives: 1–3 aspects with sentiment (positive / negative / neutral), weight, evidence keywords, and optional sub-aspects (JSON schema enforced).

```bash
uv run python src/llm_label_submit.py    # submit batch job (~$5.60 for all topics)
uv run python src/llm_label_retrieve.py  # retrieve results and write to DuckDB
```

### 18 — Aspect visualizations

Post-labeling charts: aspect sentiment distributions, trends over time (2018–2024), comparison across distance bands and hotel star ratings.

---

## Project Structure

```text
coastal-hotel/
├── scraping/                   # Playwright scrapers
│   ├── get_reviews.py          # Core review scraper (GMapsReviewsScraper)
│   ├── get_metadata.py         # Hotel metadata scraper (5 concurrent tabs)
│   ├── run.py                  # Batch runner over hotels_processed.csv
│   └── hotels_processed.csv
├── src/                        # Reusable modules + pipeline scripts
│   ├── preprocessor.py         # Preprocessor class (vi/en tokenization)
│   ├── topic_modeling.py       # BERTopic pipeline + Qdrant integration
│   ├── preprocess_to_duckdb.py # Batch preprocess → DuckDB
│   ├── embed_to_duckdb.py      # Batch embed → DuckDB
│   ├── llm_label.py            # Claude API silver-labeling logic
│   ├── llm_label_submit.py     # Submit Anthropic Batch job
│   └── llm_label_retrieve.py   # Retrieve + store batch results
├── notebooks/                  # Numbered in pipeline order
│   ├── 01_data_prepare.ipynb
│   ├── 02_google_reviews_extracting.ipynb
│   ├── 03_stay_detail.ipynb
│   ├── 05_agoda_prepare.ipynb
│   ├── 06_googlemaps_prepare.ipynb
│   ├── 07_merge_and_preprocess.ipynb
│   ├── 08_topic_prepare.ipynb
│   ├── 09_topic_implement.ipynb
│   ├── 10_topic_process.ipynb
│   ├── 11_topic_coast_time.ipynb
│   ├── 12_sql_implementation.ipynb
│   ├── 14_topic_sample_10k.ipynb
│   ├── 15_coast_wordcloud_banner.ipynb
│   ├── 16_db_inspection.ipynb
│   ├── 17_silver_label_test.ipynb
│   └── 18_aspect_visuals.ipynb
├── data/                       # gitignored — raw + processed data
│   ├── raw/                    # hotel.csv, distance2coast.csv, agoda-reviews.csv
│   ├── processed/              # final-reviews-en/vi.csv, merged CSVs
│   └── hotel_reviews.db        # DuckDB (1.9 GB)
├── checkpoints/                # gitignored — BERTopic .pkl files
├── IMG/                        # Generated visualization outputs
├── pyproject.toml
└── CLAUDE.md
```

---

## Data Flow

```
hotel.csv (8,574) ──┐
                    ├──[NB 01]──► hotel_with_distance.csv (8,574)
distance2coast.csv ─┘             hotel_filtered.csv (1,150)
                                           │
                     ┌─────────────────────┘
                     │
Google Maps scrape ──[NB 06]──► googlemaps-review-en-vi.csv (127,502)
Agoda CSV ───────────[NB 05]──► agoda-review-en-vi.csv (125,347)
                                           │
                                  [NB 07] merge + preprocess
                                           │
                          data-review-en-vi.csv (252,849)
                          final-reviews-en.csv (135,326)
                          final-reviews-vi.csv (116,756)
                                           │
                    preprocess_to_duckdb.py ──► REVIEW_TEXT_PROCESSED
                    embed_to_duckdb.py      ──► REVIEW_EMBEDDINGS (×768)
                                           │
                           [NB 11] fit 20 BERTopic models
                                           │
               ┌───────────────────────────┼───────────────────┐
               ↓                           ↓                   ↓
         TOPIC_LABELS               REVIEW_TOPICS        checkpoints/*.pkl
         (500+ topics)              (500k+ rows)         (20 models)
               │                                              │
      [NB 17] LLM labeling                        [NB 15] word clouds
               │                                              │
         TOPIC_ASPECTS                          coast_wordcloud_{en,vi}.png
               │
      [NB 18] aspect visualizations
```

---

## Key Numbers

| Metric | Value |
|---|---|
| Hotels scraped | 1,150 coastal hotels |
| Total reviews | 251,328 (en + vi) |
| Google Maps reviews | 127,502 |
| Agoda reviews | 125,347 |
| BERTopic models fitted | 20 (3 coast bands + 7 years × 2 languages) |
| Embedding dimensions | 768 (paraphrase-multilingual-mpnet-base-v2) |
| DuckDB size | ~1.9 GB |
| LLM labeling cost | ~$5.60 (Anthropic Batches API) |

---

## Notes

- CSS selectors in `scraping/get_reviews.py` are tightly coupled to Google Maps UI — any frontend update can break parsing.
- Sort order is hardcoded to Vietnamese: `aria-label='Phù hợp nhất'`.
- Scraping Google Maps may violate their Terms of Service — ensure you have proper authorization.
- The persistent `scraping/chrome_profile/` directory maintains session state; do not delete between runs.
- When fitting BERTopic, always pass an explicit `embedding_model` to avoid `KeyBERTInspired` `AttributeError`.
