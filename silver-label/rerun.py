"""
rerun.py — Rerun orchestrator

Scans outputs/evals/failed/ for reviews that did not pass evaluation,
re-analyzes them with the agent (optionally with modified strategy),
re-evaluates, and routes results accordingly.

Failed records that exceed MAX_RERUN_ATTEMPTS are flagged in rerun_log.json
for human review.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "5"))
THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.75"))
MAX_ATTEMPTS = int(os.getenv("MAX_RERUN_ATTEMPTS", "3"))

FAILED_DIR = Path(__file__).parent / "outputs" / "evals" / "failed"
ACCEPTED_DIR = Path(__file__).parent / "outputs" / "evals" / "accepted"
LOG_PATH = Path(__file__).parent / "outputs" / "rerun_log.json"

ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)

# Import agent and evaluator logic (reuse, don't duplicate)
from agent import analyze_review, format_review, SILVER_LABEL_TOOL, SYSTEM_PROMPT as ANALYST_PROMPT
from evaluate import evaluate_label, SYSTEM_PROMPT as EVAL_PROMPT


# ---------------------------------------------------------------------------
# Rerun log
# ---------------------------------------------------------------------------

def load_log() -> dict:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    return {"flagged_for_human": [], "rerun_history": []}


def save_log(log: dict):
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Single review rerun
# ---------------------------------------------------------------------------

def rerun_review(
    client: anthropic.Anthropic,
    result: dict,
    attempt: int,
) -> dict:
    """Re-analyze and re-evaluate a single review. Returns the updated result."""
    # On later attempts use a slightly more explicit prompt prefix
    original_review = result["source"]

    if attempt > 1:
        # Enrich the source with the previous failed label as context
        ef = original_review.get("edge_fields", {})
        section = ef.get("review_section", {})
        prev_label = json.dumps(result.get("silver_label", {}), ensure_ascii=False)
        section["_rerun_hint"] = f"Previous label was rejected. Re-analyze carefully. Previous: {prev_label}"

    label = analyze_review(client, original_review)

    # Re-evaluate
    tmp_result = {**result, "silver_label": label}
    eval_result = evaluate_label(client, tmp_result)

    return {
        **result,
        "silver_label": label,
        "eval": eval_result,
        "metadata": {
            **result.get("metadata", {}),
            "model": MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "attempt": attempt,
        },
    }


# ---------------------------------------------------------------------------
# Batch rerun
# ---------------------------------------------------------------------------

async def rerun_result_async(
    sem: asyncio.Semaphore,
    client: anthropic.Anthropic,
    result: dict,
    log: dict,
) -> dict:
    async with sem:
        review_id = result.get("review_id", "unknown")
        hotel_id = result.get("hotel_id", "unknown")

        # Check how many attempts have already been made
        attempts_so_far = result.get("metadata", {}).get("attempt", 1)

        if attempts_so_far >= MAX_ATTEMPTS:
            print(f"  [{review_id}] reached max attempts ({MAX_ATTEMPTS}), flagging for human review.")
            log["flagged_for_human"].append({
                "review_id": review_id,
                "hotel_id": hotel_id,
                "attempts": attempts_so_far,
                "last_score": result.get("eval", {}).get("composite", 0.0),
                "timestamp": datetime.utcnow().isoformat(),
            })
            result["eval"]["flagged"] = True
            return result

        next_attempt = attempts_so_far + 1
        try:
            updated = await asyncio.to_thread(rerun_review, client, result, next_attempt)
            passed = updated["eval"].get("passed", False)
            score = updated["eval"].get("composite", 0.0)
            print(f"  [{review_id}] attempt {next_attempt}: score={score:.2f} passed={passed}")
            log["rerun_history"].append({
                "review_id": review_id,
                "hotel_id": hotel_id,
                "attempt": next_attempt,
                "score": score,
                "passed": passed,
                "timestamp": datetime.utcnow().isoformat(),
            })
            return updated
        except Exception as exc:
            print(f"  [{review_id}] rerun error: {exc}")
            result["eval"]["rerun_error"] = str(exc)
            return result


async def process_failed_file(client: anthropic.Anthropic, failed_path: Path, log: dict):
    data = json.loads(failed_path.read_text(encoding="utf-8"))
    hotel_id = data["hotel_id"]
    results = data["results"]

    # Only rerun records that actually failed (not flagged ones)
    to_rerun = [r for r in results if not r.get("eval", {}).get("flagged")]
    if not to_rerun:
        print(f"[{hotel_id}] all records already flagged, skipping.")
        return

    print(f"[{hotel_id}] rerunning {len(to_rerun)} failed record(s)…")
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [rerun_result_async(sem, client, r, log) for r in to_rerun]
    updated = await asyncio.gather(*tasks)

    passed = [r for r in updated if r.get("eval", {}).get("passed")]
    still_failed = [r for r in updated if not r.get("eval", {}).get("passed")]

    # Move newly passed results to accepted/
    if passed:
        accepted_path = ACCEPTED_DIR / f"{hotel_id}_eval.json"
        existing = []
        if accepted_path.exists():
            existing = json.loads(accepted_path.read_text(encoding="utf-8")).get("results", [])
        accepted_data = {
            "hotel_id": hotel_id,
            "source": "rerun",
            "results": existing + passed,
        }
        accepted_path.write_text(json.dumps(accepted_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{hotel_id}] {len(passed)} newly passed → accepted/")

    # Overwrite failed file with remaining failures
    if still_failed:
        data["results"] = still_failed
        data["total"] = len(still_failed)
        failed_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{hotel_id}] {len(still_failed)} still failing → failed/")
    else:
        failed_path.unlink()
        print(f"  [{hotel_id}] all resolved, removed from failed/")


async def run(hotel_ids: list[str] | None = None):
    if not API_KEY or API_KEY == "sk-ant-your-key-here":
        raise EnvironmentError("Set ANTHROPIC_API_KEY in silver-label/.env before running.")

    client = anthropic.Anthropic(api_key=API_KEY)
    log = load_log()

    if hotel_ids:
        files = [FAILED_DIR / f"{hid}_eval.json" for hid in hotel_ids]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(FAILED_DIR.glob("*_eval.json"))

    if not files:
        print("No failed eval files found. Nothing to rerun.")
        return

    print(f"Rerunning {len(files)} failed file(s)…")
    for path in files:
        await process_failed_file(client, path, log)

    save_log(log)
    flagged = len(log["flagged_for_human"])
    if flagged:
        print(f"\n{flagged} review(s) flagged for human review → {LOG_PATH}")
    print("Rerun complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Silver-label rerun orchestrator")
    parser.add_argument("--hotels", nargs="*", metavar="ID", help="Hotel IDs to rerun.")
    args = parser.parse_args()
    asyncio.run(run(hotel_ids=args.hotels))
