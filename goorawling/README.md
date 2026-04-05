# Google Maps Scraper Suite

This project provides advanced web scraping tools for Google Maps:

**Primary Scraper (Recommended)**
- `gmaps_reviews_scraper.py` — Full Local Guide reviews scraper following exact specifications:
  - Uses Playwright for stealth-mode interaction
  - Scrolls until all reviews loaded
  - Expands all "See More" buttons
  - Parses with BeautifulSoup (Spider-like)
  - Groups reviews by place_id
  - Exports structured JSON with all metadata

**Legacy Scrapers**
- `scraper_playwright.py` — uses Playwright (basic dynamic page scraping).
- `scraper_selenium.py` — uses Selenium + ChromeDriver.

Important notes
- Scraping Google Maps may violate Google Terms of Service. Ensure you have the right to scrape and use the data, and avoid automated requests that violate site rules.
- The pages are highly dynamic and the provided scripts use heuristics to find content. Selectors may break — you may need to adapt extraction logic for reliability.

Setup (PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r "Day #1/playwright_selenium_scraper/requirements.txt"
# Playwright needs browsers installed:
python -m playwright install
```

Run (Full Reviews Scraper - Recommended)
```powershell
python "Day #1/playwright_selenium_scraper/gmaps_reviews_scraper.py" "https://www.google.com/maps/contrib/100907397351319007420/reviews?hl=vi" --output all_reviews.json
```

Run with visible browser (for debugging)
```powershell
python "Day #1/playwright_selenium_scraper/gmaps_reviews_scraper.py" "https://www.google.com/maps/contrib/100907397351319007420/reviews?hl=vi" --no-headless --output all_reviews.json
```

Run (Playwright basic example)
```powershell
python "Day #1/playwright_selenium_scraper/scraper_playwright.py" "https://www.google.com/maps/contrib/101050509697705912554" --headless --output output_playwright.json
```

Run (Selenium basic example)
```powershell
python "Day #1/playwright_selenium_scraper/scraper_selenium.py" "https://www.google.com/maps/contrib/101050509697705912554" --headless --output output_selenium.json
```

Outputs
- `gmaps_reviews_scraper.py` generates:
  - Structured JSON with metadata grouped by place_id
  - Each review includes: place_id, place_name, address, thumbnail, rating, timestamp, review_text, text_length, image_urls
  - Raw HTML saved for inspection/debugging
- Legacy scrapers produce: JSON with `user_metadata`, `reviews`, `places_reviewed`, `place_images` (best-effort)
- Raw HTML always saved alongside JSON (same name, `.html`)

Next steps (on request)
- Download and save review images locally for embeddings/vision models
- Add multi-profile parallel scraping
- Export to CSV, Parquet, or database formats
- Implement caching/resumable scraping for large profiles
