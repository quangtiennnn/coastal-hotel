"""
agent.py — Silver-label analyst agent

Reads hotel review JSON files from goorawling/outputs/,
sends each review through Claude with tool use to produce
structured silver labels, and saves results to outputs/runs/.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "5"))

OUTPUTS_DIR = Path(__file__).parent / "outputs" / "runs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DIR = Path(__file__).parent.parent / "goorawling" / "outputs"

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "review_analyst.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Tool definition — forces structured output from the model
# ---------------------------------------------------------------------------
SILVER_LABEL_TOOL: anthropic.types.ToolParam = {
    "name": "silver_label",
    "description": "Produce a structured silver label for a hotel review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "overall_sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
                "description": "Overall sentiment of the review.",
            },
            "star_rating_consistent": {
                "type": "boolean",
                "description": "True if the text sentiment aligns with the numeric star rating.",
            },
            "aspects": {
                "type": "object",
                "properties": {
                    "food": {"type": "string", "enum": ["positive", "neutral", "negative", "not_mentioned"]},
                    "service": {"type": "string", "enum": ["positive", "neutral", "negative", "not_mentioned"]},
                    "atmosphere": {"type": "string", "enum": ["positive", "neutral", "negative", "not_mentioned"]},
                    "location": {"type": "string", "enum": ["positive", "neutral", "negative", "not_mentioned"]},
                },
                "required": ["food", "service", "atmosphere", "location"],
            },
            "key_phrases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Up to 5 key phrases summarizing the review.",
                "maxItems": 5,
            },
            "language": {
                "type": "string",
                "description": "Detected language code, e.g. 'vi', 'en', 'ko'.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the label quality (0–1).",
            },
        },
        "required": ["overall_sentiment", "star_rating_consistent", "aspects", "key_phrases", "language", "confidence"],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_review(review: dict) -> str:
    """Convert a raw review dict into a plain-text prompt for the agent."""
    ef = review.get("edge_fields", {})
    meta = ef.get("metadata", {})
    section = ef.get("review_section", {})
    review_text_obj = section.get("review_text", {})
    aspect_rating = ef.get("aspect_rating") or section.get("aspect_rating", {})

    parts = [
        f"Reviewer: {ef.get('reviewer_name', 'Unknown')}",
        f"Rating: {meta.get('rating', 'N/A')}",
        f"Platform: {meta.get('platform', 'Google')}",
        f"Time: {meta.get('rating_time', 'N/A')}",
    ]
    if review_text_obj.get("text"):
        parts.append(f"Review text ({review_text_obj.get('lang', '?')}): {review_text_obj['text']}")
    else:
        parts.append("Review text: (no text provided)")
    if aspect_rating:
        parts.append(f"Aspect tags: {json.dumps(aspect_rating, ensure_ascii=False)}")
    return "\n".join(parts)


def extract_tool_use(response: anthropic.types.Message) -> dict | None:
    """Pull the tool_use block from an API response."""
    for block in response.content:
        if block.type == "tool_use" and block.name == "silver_label":
            return block.input
    return None


def get_review_id(review: dict) -> str:
    return review.get("edge_fields", {}).get("metadata", {}).get("review_id", "unknown")


# ---------------------------------------------------------------------------
# Core agent call
# ---------------------------------------------------------------------------

def analyze_review(client: anthropic.Anthropic, review: dict) -> dict:
    """Call the analyst agent for a single review. Returns the silver label dict."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[SILVER_LABEL_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": format_review(review)}],
    )
    label = extract_tool_use(response)
    if label is None:
        raise ValueError(f"Model did not call silver_label tool for review {get_review_id(review)}")
    return label


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def process_review_async(
    sem: asyncio.Semaphore,
    client: anthropic.Anthropic,
    hotel_id: str,
    review: dict,
) -> dict:
    """Wrap sync API call in a semaphore-controlled async task."""
    async with sem:
        review_id = get_review_id(review)
        try:
            label = await asyncio.to_thread(analyze_review, client, review)
            return {
                "review_id": review_id,
                "hotel_id": hotel_id,
                "source": review,
                "silver_label": label,
                "metadata": {
                    "model": MODEL,
                    "timestamp": datetime.utcnow().isoformat(),
                    "attempt": 1,
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "review_id": review_id,
                "hotel_id": hotel_id,
                "source": review,
                "silver_label": None,
                "metadata": {
                    "model": MODEL,
                    "timestamp": datetime.utcnow().isoformat(),
                    "attempt": 1,
                },
                "error": str(exc),
            }


async def process_hotel_file(client: anthropic.Anthropic, json_path: Path) -> Path:
    """Process all reviews in a single hotel JSON file and save a runs output."""
    hotel_id = json_path.stem.replace("_reviews", "")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Flatten all reviews across places
    all_reviews: list[dict] = []
    for reviews in data.get("reviews_by_place", {}).values():
        all_reviews.extend(reviews)

    if not all_reviews:
        print(f"[{hotel_id}] No reviews found, skipping.")
        return None

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_review_async(sem, client, hotel_id, r) for r in all_reviews]
    results = await asyncio.gather(*tasks)

    out_path = OUTPUTS_DIR / f"{hotel_id}_labels.json"
    out_path.write_text(
        json.dumps(
            {
                "hotel_id": hotel_id,
                "source_file": json_path.name,
                "total_reviews": len(all_reviews),
                "labeled": len([r for r in results if r["silver_label"] is not None]),
                "errors": len([r for r in results if r["error"] is not None]),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{hotel_id}] {len(results)} reviews labeled → {out_path.name}")
    return out_path


async def run(hotel_ids: list[str] | None = None, limit: int | None = None):
    """
    Main entry point.

    Args:
        hotel_ids: Optional list of hotel IDs to process (e.g. ['902', '163']).
                   If None, processes all files in SOURCE_DIR.
        limit:     Cap total number of hotels processed (useful for testing).
    """
    if not API_KEY or API_KEY == "sk-ant-your-key-here":
        raise EnvironmentError("Set ANTHROPIC_API_KEY in silver-label/.env before running.")

    client = anthropic.Anthropic(api_key=API_KEY)

    if hotel_ids:
        files = [SOURCE_DIR / f"hotel_{hid}_reviews.json" for hid in hotel_ids]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(SOURCE_DIR.glob("*_reviews.json"))

    if limit:
        files = files[:limit]

    print(f"Processing {len(files)} hotel file(s) with concurrency={CONCURRENCY}…")
    for path in files:
        await process_hotel_file(client, path)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Silver-label analyst agent")
    parser.add_argument(
        "--hotels",
        nargs="*",
        metavar="ID",
        help="Hotel IDs to process (e.g. 902 163). Defaults to all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of hotel files to process.",
    )
    args = parser.parse_args()
    asyncio.run(run(hotel_ids=args.hotels, limit=args.limit))
