"""
llm_label_retrieve.py
=====================
Retrieve silver-label batch results and write them to DuckDB
(TOPIC_LABELS + TOPIC_ASPECTS).

Safe to run any time after submit: if the batch is still processing it prints
the status and exits. Writes are idempotent (UPDATE + DELETE/INSERT).
Failed requests are listed so they can be re-submitted with
`llm_label_submit.py <run_id>` (unlabeled topics are picked up automatically).

Usage:
    # Retrieve every pending batch in batches/
    uv run python src/llm_label_retrieve.py

    # Retrieve a specific batch file
    uv run python src/llm_label_retrieve.py silver_label_20260611_120000.json

    # Block until the batch finishes (poll every 60 s)
    uv run python src/llm_label_retrieve.py --wait
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from llm_label import (
    BATCH_DIR,
    DB_PATH,
    ensure_label_tables,
    get_client,
    parse_custom_id,
    parse_label,
    write_label,
)

POLL_SECONDS = 60


def retrieve_one(client, con, record_path: Path, wait: bool) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    batch_id = record["batch_id"]

    batch = client.messages.batches.retrieve(batch_id)
    while batch.processing_status != "ended":
        c = batch.request_counts
        print(f"[{record_path.name}] {batch.processing_status} - "
              f"processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
        if not wait:
            return
        time.sleep(POLL_SECONDS)
        batch = client.messages.batches.retrieve(batch_id)

    n_ok, failed = 0, []
    for result in client.messages.batches.results(batch_id):
        rtype = result.result.type
        if rtype != "succeeded":
            failed.append((result.custom_id, rtype))
            continue
        msg = result.result.message
        try:
            text = next(b.text for b in msg.content if b.type == "text")
            label = parse_label(text)
            run_id, topic_id = parse_custom_id(result.custom_id)
            write_label(con, run_id, topic_id, label)
            n_ok += 1
        except Exception as exc:  # malformed JSON / unexpected shape
            failed.append((result.custom_id, f"parse_error: {exc}"))

    record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    record["n_succeeded"] = n_ok
    record["n_failed"] = len(failed)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[{record_path.name}] wrote {n_ok} labels, {len(failed)} failed")
    for cid, why in failed:
        print(f"  ! {cid}: {why}")
    if failed:
        runs = sorted({parse_custom_id(cid)[0] for cid, _ in failed})
        print("Re-submit failed topics with:")
        print(f"  uv run python src/llm_label_submit.py {' '.join(runs)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve silver-label batch results.")
    ap.add_argument("batch_files", nargs="*",
                    help="batch json files in batches/ (default: all pending)")
    ap.add_argument("--wait", action="store_true",
                    help=f"poll every {POLL_SECONDS}s until the batch ends")
    ap.add_argument("--redo", action="store_true",
                    help="also re-process batches already marked retrieved")
    args = ap.parse_args()

    if args.batch_files:
        paths = [BATCH_DIR / f if not Path(f).exists() else Path(f) for f in args.batch_files]
    else:
        paths = sorted(BATCH_DIR.glob("*.json"))
        if not args.redo:
            paths = [p for p in paths
                     if json.loads(p.read_text(encoding="utf-8")).get("retrieved_at") is None]

    if not paths:
        print("No pending batches found in batches/.")
        return

    client = get_client()
    con = duckdb.connect(str(DB_PATH))
    ensure_label_tables(con)
    try:
        for p in paths:
            retrieve_one(client, con, p, wait=args.wait)
    finally:
        con.close()


if __name__ == "__main__":
    main()
