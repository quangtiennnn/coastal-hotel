#!/usr/bin/env python3
"""
Batch Google Maps hotel scraper.
Reads hotel_filtered.csv, searches each hotel by name + address,
navigates to the top result, and extracts structured data.
"""

import asyncio
import datetime
import json
import re
import urllib.parse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

PROFILE_DIR  = Path(__file__).parent / "chrome_profile"
OUTPUT_DIR   = Path(__file__).parent / "outputs"
CSV_PATH     = Path(__file__).parent.parent / "data" / "hotel_filtered.csv"
GMAPS_SEARCH = "https://www.google.com/maps/search/"
CONCURRENCY  = 5


def extract_hotel_data(html: str, hotel_link: str = None) -> dict:
    """Parse Google Maps place page HTML into structured dict."""
    soup = BeautifulSoup(html, 'html.parser')
    result = {}

    # Hotel name
    hotel_el = soup.select_one('.DUwDvf.lfPIob')
    if hotel_el:
        result["hotel_name"] = hotel_el.text.strip()

    # Rating and review count
    rating_el = soup.select_one('.F7nice')
    if rating_el:
        stars_el = rating_el.select_one('.ceNzKf')
        if stars_el:
            stars_label = stars_el.get("aria-label", "")  # e.g. "4.4 stars "
            m = re.search(r'(\d+\.?\d*)', stars_label)
            if m:
                result["rating"] = float(m.group(1))

        reviews_el = rating_el.select_one('span[role="img"][aria-label*="review"]')
        if reviews_el:
            reviews_label = reviews_el.get("aria-label", "")  # e.g. "3,942 reviews"
            m = re.search(r'([\d,]+)', reviews_label)
            if m:
                result["no_reviews"] = int(m.group(1).replace(",", ""))

    # Accommodation type
    accom_el = soup.select_one('.mgr77e')
    if accom_el:
        result["accommodation"] = accom_el.text.strip()[1:]

    # Data-item-id elements (address, phone, website, etc.)
    for el in soup.select("[data-item-id][aria-label]"):
        raw_id = el.get("data-item-id", "").strip()
        aria = el.get("aria-label", "").strip()
        if not raw_id:
            continue
        key = "phone" if raw_id.split(":")[0] == "phone" else raw_id
        result[key] = aria

    # Facilities / amenities
    result["facilities"] = [el.text.strip() for el in soup.select('.gSamH') if el.text.strip()]

    # Price
    price_el = soup.select_one('.dkgw2 .fontTitleLarge.Cbys4b')
    if price_el:
        result["hotel_price"] = price_el.text.strip()

    # Restructure into metadata + informations
    metadata_keys = ["hotel_name", "rating", "no_reviews", "accommodation", "hotel_price"]
    metadata = {k: result[k] for k in metadata_keys if k in result}
    metadata["hotel_id"] = None  # filled in by caller
    metadata["hotel_link"] = hotel_link
    metadata["scrape_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    informations = {k: v for k, v in result.items() if k not in metadata_keys}

    return {"metadata": metadata, "informations": informations}


SORRY_WAIT_SECONDS = 30 * 60  # 30 minutes


async def check_and_wait_for_sorry(page):
    """If Google rate-limited us (sorry page), wait 30 minutes then reload."""
    if "google.com/sorry/index" in page.url:
        print(f"  [!] Google sorry page detected — waiting {SORRY_WAIT_SECONDS // 60} minutes...")
        await asyncio.sleep(SORRY_WAIT_SECONDS)
        await page.reload(timeout=60000)
        await page.wait_for_timeout(3000)


async def search_and_navigate(page, hotel_name: str, address: str) -> bool:
    """Search Google Maps for a hotel and navigate to the first result.
    Returns True if a result was found and navigated to."""
    query = f"{hotel_name}, {address.strip()}"
    url = GMAPS_SEARCH + urllib.parse.quote(query)
    await page.goto(url, timeout=60000)
    await check_and_wait_for_sorry(page)
    await page.wait_for_timeout(3000)

    count = await page.locator('a[class*="hfpxzc"]').count()
    if count == 0:
        # Google navigated directly to the place page — already there
        return True

    # Multiple results listed — click the first one
    await page.locator('a[class*="hfpxzc"]').first.click()
    await page.wait_for_timeout(3000)
    await asyncio.sleep(3)
    return True


async def scrape_hotel(browser_context, hotel_id: int, hotel_name: str, address: str,
                       all_results: dict, summary_path: Path,
                       sem: asyncio.Semaphore, lock: asyncio.Lock):
    """Scrape a single hotel in its own tab."""
    out_path = OUTPUT_DIR / f"hotel_{hotel_id}.json"
    if out_path.exists():
        print(f"  [{hotel_id}] Skipping — already scraped")
        return

    async with sem:
        page = await browser_context.new_page()
        try:
            print(f"\nProcessing [{hotel_id}]: {hotel_name}")
            found = await search_and_navigate(page, hotel_name, address)
            if not found:
                return

            await asyncio.sleep(3)
            html = await page.content()
            hotel_link = page.url
        finally:
            await page.close()

    data = extract_hotel_data(html, hotel_link)
    data["metadata"]["hotel_id"] = hotel_id

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out_path.name}")

    async with lock:
        all_results[str(hotel_id)] = data
        summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Summary updated → {summary_path.name}")


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    hotels = pd.read_csv(CSV_PATH, encoding="utf-8-sig")[["hotel_id", "hotel_name", "addressline1"]]
    print(f"Loaded {len(hotels)} hotels from {CSV_PATH.name}")

    summary_path = OUTPUT_DIR / "hotels_scraped.json"
    all_results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    sem  = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()

    async with async_playwright() as p:
        browser_context = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )

        tasks = [
            scrape_hotel(
                browser_context,
                int(row["hotel_id"]), str(row["hotel_name"]), str(row["addressline1"]),
                all_results, summary_path, sem, lock,
            )
            for _, row in hotels.iterrows()
        ]
        await asyncio.gather(*tasks)

        await browser_context.close()


if __name__ == "__main__":
    asyncio.run(main())
