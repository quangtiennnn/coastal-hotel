# Coastal Hotel Review Analysis

A research pipeline for scraping, processing, and analyzing Google Maps reviews of coastal hotels in Vietnam. The project spans web scraping, data preparation, topic modeling, and agentic AI evaluation.

## Project Structure

```text
coastal-hotel/
├── goorawling/               # Google Maps review scraper
│   ├── gmaps_reviews_scraper.py
│   ├── run.py
│   ├── streamlit_app.py      # Batch scraping UI
│   └── outputs/              # hotel_{id}_reviews.json files
│
├── data/                     # Datasets
│   ├── hotel.csv             # 8,574 hotels with metadata (Vietnam)
│   ├── distance2coast.csv    # 29,446 rows — hotel geospatial + coastline distance
│   ├── hotel_filtered.csv    # IQR-filtered high-engagement hotels
│   ├── hotel_with_distance.csv
│   ├── agoda-reviews-en-vi.csv
│   └── reviews_merged.csv    # Flattened Google reviews joined with hotel metadata
│
├── pre-scraping/             # Data preparation notebooks
│   ├── data-prepare.ipynb    # Merge hotel.csv + distance2coast.csv
│   ├── google-reviews-extracting.ipynb  # Flatten JSON outputs → reviews_merged.csv
│   └── extracting-data.ipynb
│
├── topic-modeling/           # BERTopic pipeline on Agoda reviews
│   ├── topic_modeling.py     # BERTopic + Qdrant + multilingual embeddings
│   └── implement.ipynb
│
├── silver-label/             # Agentic AI evaluation pipeline (Claude SDK)
│   ├── agent.py              # Review analysis agent
│   ├── evaluate.py           # Automated scoring / silver-label generation
│   └── rerun.py              # Rerun failed/low-quality runs
│
└── qdrant_storage/           # Local Qdrant vector store
```

## Setup

```bash
# Install dependencies (uses uv)
uv sync

# Install Playwright browser (first time only)
uv run playwright install chrome
```

## Data Pipeline

```text
hotel.csv ──┐
            ├─ data-prepare.ipynb ──► hotel_with_distance.csv
distance2coast.csv ──┘                        │
                                              ▼
                                   hotel_filtered.csv  (IQR filter, ~1,200 hotels)
                                              │
goorawling/outputs/hotel_{id}_reviews.json ──┤
                                              ▼
                               google-reviews-extracting.ipynb
                                              │
                                              ▼
                                    data/reviews_merged.csv
```

### Step 1 — Merge hotel metadata with coastline distances

```bash
uv run jupyter notebook pre-scraping/data-prepare.ipynb
```

### Step 2 — Filter hotels by review count (IQR outlier filter)

```bash
uv run python data/prepare.py
# Output: data/hotels_high_reviews.csv
```

### Step 3 — Scrape Google Maps reviews

```bash
# Single URL
uv run python goorawling/gmaps_reviews_scraper.py "<contributor_url>" --output output.json

# Batch via Streamlit UI
uv run streamlit run goorawling/streamlit_app.py
```

Each hotel outputs `goorawling/outputs/hotel_{hotel_id}_reviews.json`.

### Step 4 — Extract & merge reviews into CSV

```bash
uv run jupyter notebook pre-scraping/google-reviews-extracting.ipynb
# Output: data/reviews_merged.csv
```

## Modules

### `goorawling/` — Scraper

Async Playwright scraper with stealth headers. Scrolls the reviews panel, expands all "See More" buttons, and parses with BeautifulSoup.

Output JSON schema per hotel:

```json
{
  "metadata": { "source_url", "total_reviews", "timestamp" },
  "reviews_by_place": {
    "<place_id>": [
      { "reviewer_name", "rating", "rating_time", "review_text", "aspect_rating", ... }
    ]
  }
}
```

### `topic-modeling/` — BERTopic Pipeline

Multilingual topic modeling on Agoda reviews (Vietnamese + English):

- Embedding: `paraphrase-multilingual-mpnet-base-v2`
- Clustering: HDBSCAN
- Vector store: Qdrant (local, `localhost:6333`)
- Representation: KeyBERT + Maximal Marginal Relevance

```bash
uv run jupyter notebook topic-modeling/implement.ipynb
```

> Requires Qdrant running locally. Start with:
> `docker run -p 6333:6333 qdrant/qdrant`

### `silver-label/` — Agentic Evaluation

Claude Agent SDK pipeline to automatically evaluate review analysis quality:

- `agent.py` — runs review analysis agent
- `evaluate.py` — scores outputs and generates silver labels
- `rerun.py` — reprocesses failed/low-score runs

Requires `ANTHROPIC_API_KEY` in `silver-label/.env`.

## Notes

- Scraping Google Maps may violate their Terms of Service — ensure you have proper authorization.
- CSS selectors in the scraper are tightly coupled to Google Maps UI and may break on frontend updates.
- The persistent `chrome_profile/` directory maintains session state — do not delete between runs if authenticated access is needed.
