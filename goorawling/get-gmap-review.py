#!/usr/bin/env python3
"""
Google Maps Local Guide Reviews Scraper

Workflow:
1. Open page with Playwright (stealth config)
2. Scroll to load reviews
3. Expand all "See More" buttons and capture HTML
4. Parse review blocks with BeautifulSoup
5. Save results grouped by place_id

Usage:
  python gmaps_reviews_scraper.py "https://www.google.com/maps/contrib/100907397351319007420/reviews?hl=vi" --browser chrome
"""

import argparse
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup
from tqdm import tqdm
import asyncio


class NoReviewsTab(Exception):
    """Raised when the Reviews tab is not found on the page."""


class SorryPage(Exception):
    """Raised when Google rate-limit (sorry) page is detected."""


class GMapsReviewsScraper:
    def __init__(self, url: str, headless: bool = True, output: str = "gmaps_reviews.json",
                 browser: str = "chromium", context=None):
        # Ensure the URL contains a language tag (hl=vi) unless already present
        self.url = self._ensure_hl_param(url, lang='vi')
        self.headless = headless
        self.output = output
        self.browser = "chrome"
        self.html = None
        self.reviews_by_place = defaultdict(list)
        # If context is provided externally, we don't own the browser lifecycle
        self._playwright = None
        self._context = context
        self._owns_browser = context is None

    def _ensure_hl_param(self, url: str, lang: str = 'vi') -> str:
        """Return URL with `hl=<lang>` added if not already present.

        - If `hl=` is already present in the URL, return unchanged.
        - Otherwise append `?hl=lang` or `&hl=lang` appropriately.
        """
        if 'hl=' in url:
            return url
        # preserve fragments
        parts = url.split('#', 1)
        base = parts[0]
        frag = ('#' + parts[1]) if len(parts) == 2 else ''
        if '?' in base:
            return f"{base}&hl={lang}{frag}"
        else:
            return f"{base}?hl={lang}{frag}"

    async def _wait_if_sorry(self, page: Page) -> None:
        """If Google rate-limited us (sorry page), close the tab and skip."""
        if "google.com/sorry/index" in page.url:
            print("  [!] Google sorry page detected — closing tab and moving to next hotel")
            await page.close()
            raise SorryPage()

    async def step1_open_page(self) -> Page:
        """Step 1: Open Page with Playwright (stealth-like config)"""
        print("Step 1: Opening page with Playwright (persistent Chrome profile)...")

        if self._owns_browser:
            # Start playwright and create our own context
            self._playwright = await async_playwright().start()

            profile_dir = Path(__file__).parent / "chrome_profile"
            profile_dir.mkdir(exist_ok=True)

            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=self.headless,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"],
            )

            await self._context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
            """)

        page = await self._context.new_page()
        page.set_default_timeout(60000)

        print(f"Navigating to {self.url}")
        await page.goto(self.url)
        await self._wait_if_sorry(page)
        time.sleep(2)  # Wait a bit for the page to stabilize
        # If search results list is shown instead of a place page, skip
        if await page.locator("a.hfpxzc").is_visible():
            print("Search results page detected (hfpxzc) — skipping to next hotel")
            raise NoReviewsTab()

        # Click the Reviews tab (data-tab-index="2")
        try:
            reviews_tab = page.locator("button.hh2c6[data-tab-index='2']")
            await reviews_tab.wait_for(timeout=10000)
            await reviews_tab.click()
            print("Clicked Reviews tab")
        except Exception:
            print("Reviews tab not found — saving empty JSON and skipping")
            raise NoReviewsTab()

        await asyncio.sleep(2)

        # Click the sort dropdown ("Phù hợp nhất") then select next option with arrow + enter
        try:
            sort_btn = page.locator("button.HQzyZ[aria-label='Phù hợp nhất']")
            await sort_btn.wait_for(timeout=10000)
            await sort_btn.click()
            print("Clicked sort dropdown")
            await page.keyboard.press("ArrowDown")
            await page.keyboard.press("Enter")
            print("Selected next sort option")
        except Exception as e:
            print(f"Could not interact with sort dropdown: {e}")

        # Wait for main content to load
        try:
            await page.wait_for_selector("div[data-review-id]", timeout=15000)
            print("Reviews container loaded")
        except Exception as e:
            print(f"Reviews container not found immediately: {e}")

        await asyncio.sleep(2)
        return page
    
    async def step2_scroll_until_all_reviews_loaded(self, page: Page) -> None:
        """Scroll down in the results panel for specified duration"""
        duration = 60  # seconds (1 minute)
        try:
            print(f"    Scrolling results for {duration} seconds...")
            
            # Try multiple selectors for the results panel
            selectors = [
                ".m6QErb.DxyBCb.kA9KIf.XiKgde",
                "div[role='region']",
                "div[role='feed']",
                ".m6QErb",
                "div.DxyBCb"
            ]
            panel = None
            for selector in selectors:
                try:
                    elements = page.locator(selector)
                    if await elements.count() > 0:
                        panel = elements.first
                        break
                except:
                    continue
            
            if not panel:
                print("      Warning: Could not find results panel, using page scroll")
                panel = page
            
            # Hover to ensure focus
            try:
                await panel.hover()
            except:
                pass
            
            await page.wait_for_timeout(500)
            
            start_time = time.time()
            scroll_count = 0
            
            while True:
                # Use mouse wheel to scroll
                for _ in range(5):
                    try:
                        await page.mouse.wheel(0, 400)
                        scroll_count += 1
                    except:
                        pass
                
                await page.wait_for_timeout(100)
                elapsed_time = time.time() - start_time
                
                if elapsed_time > duration:
                    break
            
            print(f"      Scrolled {scroll_count} times")
            
        except Exception as e:
            print(f"    Error scrolling: {str(e)}")


    async def step3_expand_see_more_buttons(self, page: Page) -> None:
        """Step 3: Expand All 'See More' Review Text Buttons and capture HTML"""
        print("\nStep 3: Expanding 'See More' buttons (selector: button.w8nwRe.kyuRq)...")
        
        expansion_count = 0

        # Use tqdm to show progress; total is unknown so we use an indeterminate bar
        pbar = tqdm(desc='Expanding reviews', unit='btn')

        # Keep clicking until no more buttons remain (DOM updates after each click)
        while True:
            buttons = await page.locator("button.w8nwRe.kyuRq").all()

            if not buttons:
                pbar.close()
                print("No more 'See More' buttons found")
                break

            try:
                # Always click the first button in the updated list
                btn = buttons[0]
                await btn.click()

                expansion_count += 1
                pbar.update(1)

                # Wait for expanded content to render
                # await asyncio.sleep(0.01)

            except Exception as e:
                pbar.close()
                print(f"  Could not expand button: {e}")
                break

        print(f"Expanded {expansion_count} 'See More' buttons total")
        
        # Capture HTML after expansion
        print("Capturing page HTML...")
        self.html = await page.content()
        print(f"Captured {len(self.html)} bytes of HTML")

    def step4_parse_reviews(self) -> None:
        """Step 4: Spider - Parse All Review Blocks"""
        print("\nStep 4: Parsing review blocks with BeautifulSoup...")
        
        if not self.html:
            print("ERROR: No HTML to parse. HTML should be captured in step3.")
            return
        
        soup = BeautifulSoup(self.html, "html.parser")
        
        # Find all review blocks: <div data-review-id="...">
        review_blocks = soup.find_all("div", attrs={"data-review-id": True})
        print(f"Found {len(review_blocks)} review blocks (before dedup)")

        seen_review_ids = set()
        parsed = 0
        skipped = 0

        for block in review_blocks:
            review_data = self._extract_review_metadata(block)
            if not review_data:
                continue

            review_id = review_data.get("edge_fields", {}).get("metadata", {}).get("review_id")
            if review_id:
                if review_id in seen_review_ids:
                    skipped += 1
                    continue
                seen_review_ids.add(review_id)

            place_id = review_data.get("place_id", "unknown")
            self.reviews_by_place[place_id].append(review_data)
            parsed += 1
            if parsed % 10 == 0:
                print(f"  Parsed {parsed} reviews...")

        print(f"Extraction complete. Parsed: {parsed}, Duplicates skipped: {skipped}, Total places: {len(self.reviews_by_place)}")

    def _extract_review_metadata(self, block) -> Dict[str, Any]:
        """Extract place + review metadata from a review block"""
        review_data = {
            "place_node": {},
            "edge_fields": {}
        }

        try:
            # --- PLACE NODE FIELDS ---

            # Extract place_id from place URL
            place_link = block.find("a", href=re.compile(r"cid=|!1s"))
            if place_link:
                href = place_link.get("href", "")
                cid_match = re.search(r"cid=(\d+)", href)
                if cid_match:
                    review_data["place_node"]["place_id"] = cid_match.group(1)
                else:
                    place_id_match = re.search(r"!1s([a-zA-Z0-9]+)", href)
                    if place_id_match:
                        review_data["place_node"]["place_id"] = place_id_match.group(1)

            # reviewer_name: div.d4r55
            reviewer_name = block.find("div", class_="d4r55")
            if reviewer_name:
                review_data["edge_fields"]["reviewer_name"] = reviewer_name.get_text(strip=True)

            # reviewer_info: div.RfnDt
            reviewer_info = block.find("div", class_="RfnDt")
            if reviewer_info:
                review_data["edge_fields"]["reviewer_info"] = reviewer_info.get_text(strip=True)

            # thumbnail_image: img.WEBjve
            thumbnail_img = block.find("img", class_="WEBjve")
            if thumbnail_img:
                review_data["place_node"]["thumbnail_image"] = thumbnail_img.get("src", "")

            # image_urls: button.Tya61d background-image style
            image_urls = []
            for btn in block.find_all("button", class_="Tya61d"):
                style = btn.get("style", "")
                url_match = re.search(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
                if url_match:
                    image_urls.append(url_match.group(1))
            if image_urls:
                review_data["edge_fields"]["image_urls"] = image_urls

            # --- METADATA (div.DU9Pgb) ---
            metadata = {}

            # review_id: data-review-id on the block itself
            review_id = block.get("data-review-id")
            if review_id:
                metadata["review_id"] = review_id

            du9pgb = block.find("div", class_="DU9Pgb")
            if du9pgb:
                # rating: span.fontBodyLarge.fzvQIb → "5/5"
                rating_el = du9pgb.find("span", class_="fzvQIb")
                if rating_el:
                    metadata["rating"] = rating_el.get_text(strip=True)

                # rating_time + platform: span.xRkPPb
                xrk = du9pgb.find("span", class_="xRkPPb")
                if xrk:
                    # rating_time is the direct text node before any child span
                    time_parts = [t for t in xrk.strings
                                  if t.parent == xrk and t.strip()]
                    if time_parts:
                        metadata["rating_time"] = time_parts[0].strip()

                    # platform: span.qmhsmd text
                    platform_el = xrk.find("span", class_="qmhsmd")
                    if platform_el:
                        metadata["platform"] = platform_el.get_text(strip=True)

            if metadata:
                review_data["edge_fields"]["metadata"] = metadata

            # --- REVIEW SECTION (div.MyEned) ---
            review_section = {}

            my_ened = block.find("div", class_="MyEned")
            if my_ened:
                # review_text: {text, lang}
                text_span = my_ened.find("span", class_="wiI7pd")
                review_section["review_text"] = {
                    "text": text_span.get_text(strip=True) if text_span else "",
                    "lang": my_ened.get("lang", ""),
                }

                # aspect_rating: each div.PBK6be → {original Vietnamese key: value string}
                aspect_rating = {}
                for pbk in my_ened.find_all("div", class_="PBK6be"):
                    # Case 1: inline <b>Label:</b> value  (e.g. <b>Phòng:</b> 5)
                    b_tag = pbk.find("b")
                    if b_tag:
                        key = b_tag.get_text(strip=True).rstrip(":")
                        # Value is the remaining text in the same span after the <b>
                        parent_span = b_tag.parent
                        raw = parent_span.get_text(" ", strip=True)
                        value = raw.replace(b_tag.get_text(strip=True), "", 1).strip(" :")
                        if key and value:
                            aspect_rating[key] = value
                        continue

                    # Case 2: bold-style span label + separate value div
                    rfds = pbk.find_all("span", class_="RfDO5c")
                    if len(rfds) >= 1:
                        # Key span: contains a bold-styled inner span
                        key_span = rfds[0].find("span", style=re.compile(r"font-weight\s*:\s*bold"))
                        if key_span:
                            key = key_span.get_text(strip=True)
                            # Value: text from the second RfDO5c, first non-empty span
                            value = ""
                            if len(rfds) >= 2:
                                for s in rfds[1].find_all("span"):
                                    txt = s.get_text(strip=True)
                                    if txt:
                                        value = txt
                                        break
                            if key and value:
                                aspect_rating[key] = value

                if aspect_rating:
                    review_section["aspect_rating"] = aspect_rating

            # --- HOTEL RESPOND (div.CDe7pd) ---
            cde7pd = block.find("div", class_="CDe7pd")
            if cde7pd:
                respond_el = cde7pd.find("div", class_="wiI7pd")
                if respond_el:
                    review_section["hotel_respond"] = {
                        "text": respond_el.get_text(strip=True),
                        "lang": respond_el.get("lang", ""),
                    }

            if review_section:
                review_data["edge_fields"]["review_section"] = review_section

        except Exception as e:
            print(f"  Error extracting review metadata: {e}")

        return review_data if (review_data["place_node"] or review_data["edge_fields"]) else None

    def save_json(self) -> None:
        """Save reviews grouped by place_id as JSON"""
        print(f"\nSaving {len(self.reviews_by_place)} places to {self.output}...")
        
        output_data = {
            "metadata": {
                "source_url": self.url,
                "total_places": len(self.reviews_by_place),
                "total_reviews": sum(len(reviews) for reviews in self.reviews_by_place.values()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "reviews_by_place": dict(self.reviews_by_place)
        }
        
        with open(self.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to {self.output}")
        
        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Total places reviewed: {output_data['metadata']['total_places']}")
        print(f"Total reviews: {output_data['metadata']['total_reviews']}")
        for place_id, reviews in list(self.reviews_by_place.items())[:5]:
            print(f"  - {place_id}: {len(reviews)} reviews")
        if len(self.reviews_by_place) > 5:
            print(f"  ... and {len(self.reviews_by_place) - 5} more places")

    async def run(self) -> None:
        """Execute full scraping workflow"""
        print("=" * 60)
        print("GOOGLE MAPS REVIEWS SCRAPER - Full Workflow")
        print("=" * 60)
        
        page = None
        try:
            page = await self.step1_open_page()
        except (SorryPage, NoReviewsTab) as e:
            if isinstance(e, NoReviewsTab):
                self.save_json()  # saves empty reviews_by_place
            # Close only the page; context stays alive if owned externally
            if page:
                try:
                    await page.close()
                except:
                    pass
            await self._shutdown_browser()
            return

        try:
            await self.step2_scroll_until_all_reviews_loaded(page)
            await self.step3_expand_see_more_buttons(page)
            self.step4_parse_reviews()
            self.save_json()
        finally:
            try:
                await page.close()
            except:
                pass
            await self._shutdown_browser()
            print("=" * 60)
            print("SCRAPING COMPLETE")
            print("=" * 60)

    async def _shutdown_browser(self) -> None:
        """Close context and stop playwright only if we own them."""
        if not self._owns_browser:
            return
        if self._context:
            try:
                await self._context.close()
            except:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape all reviews from Google Maps Local Guide profile"
    )
    parser.add_argument(
        "url",
        help="Local Guide profile URL (e.g., https://www.google.com/maps/contrib/...)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)"
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Disable headless mode (show browser)"
    )
    parser.add_argument(
        "--output",
        default="gmaps_reviews.json",
        help="Output JSON file (default: gmaps_reviews.json)"
    )
    parser.add_argument(
        "--browser",
        choices=["chrome"],
        default="chrome",
        help="Browser engine/channel to use. Only 'chrome' is supported. Run 'playwright install chrome'."
    )
    
    args = parser.parse_args()
    
    scraper = GMapsReviewsScraper(
        url=args.url,
        headless=args.headless,
        output=args.output,
        browser=args.browser
    )
    asyncio.run(scraper.run())
