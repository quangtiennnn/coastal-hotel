"""
llm_label_submit.py
===================
Submit silver-label requests to the Anthropic Message Batches API.

One request per (run_id, topic_id); all requested runs go into a single batch.
The batch id is persisted to batches/<name>.json so retrieval can happen later
(results stay available server-side for 29 days) - the PC can go offline.

Usage:
    # Label every current run (run_ids with a checkpoint in checkpoints/)
    uv run python src/llm_label_submit.py --all

    # Label specific runs
    uv run python src/llm_label_submit.py coast_band_A_en year_2024_vi

    # Re-label topics that already have labels
    uv run python src/llm_label_submit.py --all --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import duckdb

from llm_label import (
    BATCH_DIR,
    DB_PATH,
    MODEL,
    build_request,
    current_run_ids,
    ensure_label_tables,
    fetch_topics,
    get_client,
    is_language_run,
    sample_docs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit silver-label batch to Claude.")
    ap.add_argument("run_ids", nargs="*", help="run_ids to label (e.g. coast_band_A_en)")
    ap.add_argument("--all", action="store_true",
                    help="label all current runs (checkpointed run_ids)")
    ap.add_argument("--force", action="store_true",
                    help="also re-label topics that already have labels")
    ap.add_argument("--name", default=None,
                    help="batch file name (default: timestamp)")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))
    ensure_label_tables(con)

    if args.all:
        run_ids = current_run_ids(con)
    elif args.run_ids:
        bad = [r for r in args.run_ids if not is_language_run(r)]
        if bad:
            ap.error(
                f"mixed-language run_ids are not part of the analysis: {bad}. "
                "Use the per-language variants (e.g. coast_band_A_en, coast_band_A_vi)."
            )
        run_ids = args.run_ids
    else:
        ap.error("give run_ids or --all")

    print(f"Runs to label: {len(run_ids)}")

    requests = []
    for run_id in run_ids:
        topics = fetch_topics(con, run_id, only_unlabeled=not args.force)
        for topic_id, top_words in topics:
            docs = sample_docs(con, run_id, topic_id)
            requests.append(build_request(run_id, topic_id, top_words, docs))
        print(f"  {run_id}: {len(topics)} topics")
    con.close()

    if not requests:
        print("Nothing to label (all topics already labeled - use --force to redo).")
        return

    print(f"\nSubmitting {len(requests)} requests to {MODEL} via Batches API ...")
    client = get_client()
    batch = client.messages.batches.create(requests=requests)

    BATCH_DIR.mkdir(exist_ok=True)
    name = args.name or datetime.now().strftime("silver_label_%Y%m%d_%H%M%S")
    record = {
        "batch_id": batch.id,
        "model": MODEL,
        "n_requests": len(requests),
        "run_ids": run_ids,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_at": None,
    }
    out = BATCH_DIR / f"{name}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Batch id : {batch.id}")
    print(f"Status   : {batch.processing_status}")
    print(f"Saved    : {out}")
    print("\nRetrieve later with:")
    print(f"  uv run python src/llm_label_retrieve.py {out.name}")


if __name__ == "__main__":
    main()
