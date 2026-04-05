#!/usr/bin/env python3
"""
Batch runner for get-gmap-review.py.

Reads hotels_processed.csv, iterates each hotel by hotel_id + hotel_link,
runs the scraper, and saves results immediately after each hotel.

Per-hotel output : goorawling/outputs/hotel_{hotel_id}_reviews.json
Summary file     : goorawling/outputs/all_hotels_reviews.json
"""

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

# Import from hyphenated filename via importlib
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "get_gmap_review",
    Path(__file__).parent / "get-gmap-review.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GMapsReviewsScraper = _mod.GMapsReviewsScraper

CSV_PATH     = Path(__file__).parent / "hotels_processed.csv"
OUTPUT_DIR   = Path(__file__).parent / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "all_hotels_reviews.json"
PROFILE_DIR  = Path(__file__).parent / "chrome_profile"


async def scrape_hotel(context, hotel_id: int, hotel_name: str, hotel_link: str,
                       summary: dict) -> None:
    out_path = OUTPUT_DIR / f"hotel_{hotel_id}_reviews.json"

    if out_path.exists():
        print(f"  [{hotel_id}] Skipping — already scraped")
        return

    print(f"\n{'=' * 60}")
    print(f"[{hotel_id}] {hotel_name}")
    print(f"{'=' * 60}")

    scraper = GMapsReviewsScraper(
        url=hotel_link,
        headless=False,
        output=str(out_path),
        context=context,           # reuse the shared browser context
    )

    try:
        await scraper.run()
    except Exception as e:
        print(f"  [{hotel_id}] ERROR: {e}")
        return

    # Load the just-saved file and merge into summary immediately
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["hotel_id"]   = hotel_id
        data["hotel_name"] = hotel_name
        summary[str(hotel_id)] = data
        SUMMARY_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [{hotel_id}] Summary updated → {SUMMARY_PATH.name}")
    except Exception as e:
        print(f"  [{hotel_id}] Could not update summary: {e}")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    hotels = pd.read_csv(CSV_PATH, encoding="utf-8-sig")[
        ["hotel_id", "hotel_name", "hotel_link"]
    ].dropna(subset=["hotel_link"])

    print(f"Loaded {len(hotels)} hotels from {CSV_PATH.name}")

    summary: dict = (
        json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if SUMMARY_PATH.exists()
        else {}
    )

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
        """)

        for _, row in hotels.iterrows():
            await scrape_hotel(
                context=context,
                hotel_id=int(row["hotel_id"]),
                hotel_name=str(row["hotel_name"]),
                hotel_link=str(row["hotel_link"]),
                summary=summary,
            )

        await context.close()

    print(f"\nAll done. {len(summary)} hotels in summary.")


if __name__ == "__main__":
    asyncio.run(main())
