# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Google Maps Local Guide review scraper and dataset for analyzing coastal hotel proximity. The project integrates web scraping, data preparation, and interactive UI analysis:
- **Web scraping**: Async Playwright scraper to extract reviews from Google Maps contributor profiles
- **Data pipeline**: CSV datasets with hotel metadata, geospatial distance calculations, and statistical filtering
- **Interactive analysis**: Jupyter notebooks and Streamlit UI for data exploration and batch processing

## Package Manager & Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for package management with Python 3.14.

```bash
# Install dependencies
uv sync

# Install Playwright browsers (first-time setup)
uv run playwright install chrome
```

## Common Development Commands

### Scraper CLI
```bash
# Single URL scraping
uv run python goorawling/gmaps_reviews_scraper.py "https://www.google.com/maps/contrib/100907397351319007420/reviews?hl=vi" --output output.json

# With visible browser (debugging)
uv run python goorawling/gmaps_reviews_scraper.py "<URL>" --no-headless --output output.json
```

### Interactive UI
```bash
# Run Streamlit batch processor
uv run streamlit run goorawling/streamlit_app.py
```

### Data Preparation
```bash
# Filter hotels by review count (IQR-based)
uv run python data/prepare.py

# Run data merge notebook (interactive)
uv run jupyter notebook pre-scraping/data-prepare.ipynb
```

## Architecture

### Scraper Module (`goorawling/`)

**`gmaps_reviews_scraper.py`** — Core async scraper with 4-step pipeline:
1. Launch persistent Chrome with stealth headers (`--disable-blink-features=AutomationControlled`)
2. Spoof `navigator.webdriver` and wait for reviews container
3. Scroll reviews panel to trigger lazy loading
4. Click all "See More" buttons to expand truncated text
5. Parse expanded HTML with BeautifulSoup into structured JSON

Output schema:
```json
{
  "metadata": { "source_url", "total_places", "total_reviews", "timestamp" },
  "reviews_by_place": { "<place_id>": [ { "review_id", "rating", "timestamp", "text", "images", ... } ] }
}
```

Supports Vietnamese aspect ratings (food, service, atmosphere) and contextual metadata (meal type, price range).

**`streamlit_app.py`** — Web UI wrapping the scraper. Accepts contributor URLs or IDs (one per line or file upload). Configurable: headless mode, browser channel (chrome/chromium/msedge), timeout. Outputs scraping results to `goorawling/outputs/`.

**`chrome_profile/`** — Persistent browser profile directory maintaining cookies/login state. Do not delete between scraper runs if authenticated access required.

### Data Module (`data/`)

**`hotel.csv`** — 8,574 rows, 41 columns: hotel name, location (lat/lon), star rating, review counts, pricing (Vietnam-focused).

**`distance2coast.csv`** — 29,446 rows, 6 columns: hotel_id, coordinates (WKT format), distance to coastline in meters.

**`prepare.py`** — Filters hotels by review count using IQR statistics:
- Loads `hotel.csv`
- Calculates Q1, Q3, IQR on `number_of_reviews`
- Filters: `number_of_reviews > Q3 + 1.5*IQR`
- Outputs: `hotels_high_reviews.csv` (typically ~1,200 rows)

### Data Preparation Module (`pre-scraping/`)

**`data-prepare.ipynb`** — Merges `hotel.csv` with `distance2coast.csv` on `hotel_id`:
- Deduplicates if `distance2coast.csv` has many-to-one relationships
- Drops redundant columns (longitude/latitude)
- Left join to preserve all hotels (8,574 rows)
- Output: enriched dataset with distance-to-coast metrics

**`get-data.py`** — Helper to scrape hotel metadata from Google Maps place pages (HTML extraction).

**`hotel_review_analysis.py`** — Reusable class-based analyzer for IQR statistics and filtering (used by `data/prepare.py`).

## Data Pipeline Workflow

1. **Source data**: `hotel.csv` (metadata) + `distance2coast.csv` (geospatial)
2. **Merge** (optional): Run `pre-scraping/data-prepare.ipynb` → `hotel_with_distance.csv`
3. **Filter**: Run `data/prepare.py` → `hotels_high_reviews.csv` (high-engagement hotels only)
4. **Scrape**: Use CLI or Streamlit UI to extract reviews from contributor profiles → JSON outputs

## Scraper Fragility Notes

- CSS selectors and DOM heuristics tightly coupled to Google Maps UI — frontend changes break parsing
- Scraping Google Maps may violate Terms of Service — ensure proper authorization
- Persistent chrome profile helps with session continuity but is not a guarantee against detection
- Rendering delays handled with `wait_for_selector` timeouts; increase if pages load slowly on your network
