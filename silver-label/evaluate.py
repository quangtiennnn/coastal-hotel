"""
evaluate.py — Evaluator agent

Reads silver-label run outputs from outputs/runs/,
scores each label using a second Claude call,
and writes results to outputs/evals/.

Passed labels go to outputs/evals/accepted/
Failed labels go to outputs/evals/failed/  (picked up by rerun.py)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

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
THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.75"))

RUNS_DIR = Path(__file__).parent / "outputs" / "runs"
EVALS_DIR = Path(__file__).parent / "outputs" / "evals"
ACCEPTED_DIR = EVALS_DIR / "accepted"
FAILED_DIR = EVALS_DIR / "failed"

for d in (ACCEPTED_DIR, FAILED_DIR):
    d.mkdir(parents=True, exist_ok=True)

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "evaluator.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------
EVAL_TOOL: anthropic.types.ToolParam = {
    "name": "eval_result",
    "description": "Score a silver label against the original review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "accuracy": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Sentiment matches star rating and text.",
            },
            "completeness": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "All aspects mentioned in the review are captured.",
            },
            "consistency": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Label is internally non-contradictory.",
            },
            "faithfulness": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "No hallucinated information not in source review.",
            },
            "reasons": {
                "type": "object",
                "description": "Brief reason for each dimension score.",
                "properties": {
                    "accuracy": {"type": "string"},
                    "completeness": {"type": "string"},
                    "consistency": {"type": "string"},
                    "faithfulness": {"type": "string"},
                },
                "required": ["accuracy", "completeness", "consistency", "faithfulness"],
            },
        },
        "required": ["accuracy", "completeness", "consistency", "faithfulness", "reasons"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_eval_input(result: dict) -> str:
    """Build the evaluation prompt from a labeled result record."""
    ef = result["source"].get("edge_fields", {})
    meta = ef.get("metadata", {})
    section = ef.get("review_section", {})
    review_text = section.get("review_text", {}).get("text", "(no text)")
    rating = meta.get("rating", "N/A")
    label = result["silver_label"]

    return (
        f"--- ORIGINAL REVIEW ---\n"
        f"Rating: {rating}\n"
        f"Text: {review_text}\n\n"
        f"--- SILVER LABEL ---\n"
        f"{json.dumps(label, ensure_ascii=False, indent=2)}\n\n"
        "Please evaluate the silver label against the original review."
    )


def extract_tool_use(response: anthropic.types.Message) -> dict | None:
    for block in response.content:
        if block.type == "tool_use" and block.name == "eval_result":
            return block.input
    return None


# ---------------------------------------------------------------------------
# Core evaluator call
# ---------------------------------------------------------------------------

def evaluate_label(client: anthropic.Anthropic, result: dict) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[EVAL_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": format_eval_input(result)}],
    )
    scores = extract_tool_use(response)
    if scores is None:
        raise ValueError("Model did not call eval_result tool.")

    dimensions = {k: scores[k] for k in ("accuracy", "completeness", "consistency", "faithfulness")}
    composite = round(mean(dimensions.values()), 4)
    return {
        "scores": dimensions,
        "reasons": scores.get("reasons", {}),
        "composite": composite,
        "passed": composite >= THRESHOLD,
        "threshold": THRESHOLD,
        "model": MODEL,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def evaluate_result_async(
    sem: asyncio.Semaphore,
    client: anthropic.Anthropic,
    result: dict,
) -> dict:
    async with sem:
        if result.get("silver_label") is None:
            # Skip errored agent runs — they can't be evaluated
            result["eval"] = {"passed": False, "composite": 0.0, "error": "no silver label (agent error)"}
            return result
        try:
            result["eval"] = await asyncio.to_thread(evaluate_label, client, result)
        except Exception as exc:
            result["eval"] = {"passed": False, "composite": 0.0, "error": str(exc)}
        return result


async def evaluate_run_file(client: anthropic.Anthropic, run_path: Path):
    """Evaluate all labeled results in a single run file."""
    data = json.loads(run_path.read_text(encoding="utf-8"))
    hotel_id = data["hotel_id"]
    results = data["results"]

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [evaluate_result_async(sem, client, r) for r in results]
    evaluated = await asyncio.gather(*tasks)

    passed = [r for r in evaluated if r.get("eval", {}).get("passed")]
    failed = [r for r in evaluated if not r.get("eval", {}).get("passed")]

    summary = {
        "hotel_id": hotel_id,
        "source_run": run_path.name,
        "total": len(evaluated),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / len(evaluated), 4) if evaluated else 0,
        "results": evaluated,
    }

    # Write to accepted or failed depending on majority outcome
    out_dir = ACCEPTED_DIR if len(passed) >= len(failed) else FAILED_DIR
    out_path = out_dir / f"{hotel_id}_eval.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # If mixed: also write individual failed records to failed/ for rerun
    if passed and failed:
        failed_path = FAILED_DIR / f"{hotel_id}_eval.json"
        failed_summary = {**summary, "results": failed}
        failed_path.write_text(json.dumps(failed_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[{hotel_id}] evaluated {len(evaluated)} → "
        f"passed={len(passed)}, failed={len(failed)} → {out_path.parent.name}/"
    )


async def run(hotel_ids: list[str] | None = None):
    """Evaluate all (or selected) run files."""
    if not API_KEY or API_KEY == "sk-ant-your-key-here":
        raise EnvironmentError("Set ANTHROPIC_API_KEY in silver-label/.env before running.")

    client = anthropic.Anthropic(api_key=API_KEY)

    if hotel_ids:
        files = [RUNS_DIR / f"{hid}_labels.json" for hid in hotel_ids]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(RUNS_DIR.glob("*_labels.json"))

    if not files:
        print("No run files found. Run agent.py first.")
        return

    print(f"Evaluating {len(files)} run file(s)…")
    for path in files:
        await evaluate_run_file(client, path)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Silver-label evaluator agent")
    parser.add_argument("--hotels", nargs="*", metavar="ID", help="Hotel IDs to evaluate.")
    args = parser.parse_args()
    asyncio.run(run(hotel_ids=args.hotels))
