# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based pipeline for scraping and analyzing Google Maps hotel reviews, focused on Vietnamese coastal hotels. The stack covers async Playwright scraping, NLP/topic modeling (BERTopic, spaCy, sentence-transformers), vector storage (Qdrant), and Claude API integration.

## Package Manager & Environment

Uses [`uv`](https://docs.astral.sh/uv/) with Python 3.14.

```bash
uv sync
uv run playwright install chrome   # first-time browser setup
```

## Common Development Commands

```bash
# Scrape reviews for a single Google Maps contributor profile
uv run python goorawling/get-gmap-review.py "https://www.google.com/maps/contrib/<id>/reviews" --output output.json

# With visible browser (debugging)
uv run python goorawling/get-gmap-review.py "<URL>" --no-headless --output output.json

# Batch-scrape reviews for all hotels in goorawling/hotels_processed.csv
uv run python goorawling/run.py

# Batch-scrape hotel metadata from Google Maps search results
uv run python pre-scraping/get-data.py   # reads data/hotel_filtered.csv
```

## Architecture

```text
coastal-hotel/
├── scraping/               # all Playwright-based scrapers
│   ├── get_reviews.py      # core scraper: GMapsReviewsScraper class
│   ├── get_metadata.py     # hotel metadata scraper (5 concurrent tabs)
│   ├── run.py              # batch runner over hotels_processed.csv
│   ├── hotels_processed.csv
│   └── chrome_profile/     # persistent Chrome session (gitignored)
├── src/                    # reusable Python modules
│   ├── preprocessor.py     # Preprocessor class (vi/en tokenization)
│   └── topic_modeling.py   # BERTopic pipeline + Qdrant integration
├── notebooks/              # numbered in pipeline order
│   ├── 01_data_prepare.ipynb
│   ├── 02_google_reviews_extracting.ipynb
│   ├── 03_stay_detail.ipynb
│   ├── 04_preprocess.ipynb
│   ├── 05_agoda_prepare.ipynb
│   ├── 06_googlemaps_prepare.ipynb
│   ├── 07_merge_and_preprocess.ipynb
│   ├── 08_topic_prepare.ipynb
│   ├── 09_topic_implement.ipynb
│   └── 10_topic_process.ipynb
├── data/                   # gitignored — raw + processed data
│   ├── raw/                # source CSVs (hotel.csv, distance2coast.csv, etc.)
│   └── processed/          # cleaned outputs (final-reviews-en/vi.csv, etc.)
├── qdrant_storage/         # local Qdrant DB — gitignored, regenerable
├── pyproject.toml
└── CLAUDE.md
```

### Scraper Module (`scraping/`)

**`get_reviews.py`** — Core scraper; defines `GMapsReviewsScraper` class with a 4-step async pipeline:

1. Open persistent Chrome (stealth: `--disable-blink-features=AutomationControlled`, spoofed `navigator.webdriver`), navigate to contributor profile, click Reviews tab, set sort order
2. Scroll review panel for 60 seconds to trigger lazy-loading
3. Click all `button.w8nwRe.kyuRq` ("See More") buttons, then capture full page HTML
4. Parse HTML with BeautifulSoup: extracts `place_id`, rating, timestamp, review text, aspect ratings (Vietnamese: food/service/atmosphere), hotel responses, image URLs

The class accepts an optional external `context` (shared browser session); when provided, it does not own the browser lifecycle. Always appends `hl=vi` to URLs if absent.

Output schema:

```json
{
  "metadata": { "source_url", "total_places", "total_reviews", "timestamp" },
  "reviews_by_place": { "<place_id>": [ { "place_node", "edge_fields": { "metadata", "review_section", "image_urls" } } ] }
}
```

Exceptions: `NoReviewsTab` (saves empty JSON, continues), `SorryPage` (closes tab, skips hotel).

**`run.py`** — Batch runner. Reads `scraping/hotels_processed.csv` (`hotel_id`, `hotel_name`, `hotel_link` columns), launches one shared persistent Chrome context, and calls `GMapsReviewsScraper` for each hotel sequentially. Skips hotels whose output file already exists. Saves per-hotel JSON to `scraping/outputs/hotel_{id}_reviews.json` and maintains `scraping/outputs/all_hotels_reviews.json` summary.

**`get_metadata.py`** — Batch scraper for hotel metadata. Reads `data/hotel_filtered.csv`, searches each hotel on Google Maps, and parses HTML for: name, rating, review count, accommodation type, phone, address, facilities, price. Runs up to 5 concurrent tabs (semaphore).

### Source Module (`src/`)

**`preprocessor.py`** — `Preprocessor` class. Normalizes and word-segments review text by language: ViTokenizer for Vietnamese (joins compound words with underscores), spaCy for English. Operates on `df["review_text"]` + `df["language"]`, adds `processed_text` column.

**`topic_modeling.py`** — Full BERTopic pipeline: loads reviews, encodes with `paraphrase-multilingual-mpnet-base-v2`, stores vectors in Qdrant, runs BERTopic with UMAP + HDBSCAN + KeyBERT representation.

### Key Dependencies

- **BERTopic + UMAP + HDBSCAN** — topic modeling on review text
- **sentence-transformers** — text embeddings
- **spaCy + pyvi** — Vietnamese NLP tokenization
- **qdrant-client** — vector database (local at `qdrant_storage/`)
- **anthropic** — Claude API integration

## Scraper Fragility Notes

- CSS class selectors (`button.hh2c6[data-tab-index='2']`, `button.w8nwRe.kyuRq`, `div[data-review-id]`, etc.) are tightly coupled to Google Maps UI — any frontend update breaks parsing
- Hardcoded to Chrome channel only (`channel="chrome"`); `playwright install chrome` is required
- Sort dropdown label is hardcoded to Vietnamese: `aria-label='Phù hợp nhất'`
- Google rate-limiting (`/sorry/index`): `get_reviews.py` skips the hotel; `get_metadata.py` waits 30 minutes then retries
