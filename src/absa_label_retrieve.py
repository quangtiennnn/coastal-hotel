"""
absa_label_retrieve.py
======================
Retrieve ABSA batch results and write them to DuckDB
(REVIEW_ASPECTS + REVIEW_ASPECT_ROLLUP).

Per aspect row this:
  1. snake_cases the free-generated sub_aspect (raw preserved in sub_aspect_raw)
  2. DERIVES key_aspect via the sub->key map (unmapped -> 'other')
  3. validates evidence is a verbatim substring of the review
     (stored in evidence_valid - the hallucination check)

Safe to run any time after submit: if the batch is still processing it prints
the status and exits. Writes are idempotent (DELETE + INSERT per review).
Reviews in failed/unparseable chunks stay unlabeled, so a plain re-run of
absa_label_submit.py picks them up.

Usage:
    uv run python src/absa_label_retrieve.py                   # all pending
    uv run python src/absa_label_retrieve.py absa_label_x.json # one batch
    uv run python src/absa_label_retrieve.py --wait            # poll to done
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from absa_label import (
    BATCH_DIR,
    DB_PATH,
    alias_map,
    ensure_absa_tables,
    fetch_review_texts,
    get_client,
    normalize_aspects,
    parse_batch_result,
    write_review_aspects,
)

POLL_SECONDS = 60


def retrieve_one(client, get_con,
                 record_path: Path, wait: bool) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("kind") != "absa_review_label":
        return  # not ours (e.g. topic-level silver_label batches)
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

    # Batch ended - only NOW do we need the DB write lock
    con = get_con()

    manifest: dict[str, list[str]] = record.get("chunks", {})
    all_ids = [rid for ids in manifest.values() for rid in ids]
    texts = fetch_review_texts(con, all_ids)

    n_reviews, n_aspects, n_bad_evidence, failed = 0, 0, 0, []
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type != "succeeded":
            failed.append((cid, result.result.type))
            continue
        msg = result.result.message
        if msg.stop_reason == "max_tokens":
            failed.append((cid, "max_tokens: output truncated - raise "
                                "MAX_TOKENS or lower N_REVIEWS_PER_REQUEST"))
            continue
        try:
            text = next(b.text for b in msg.content if b.type == "text")
            reviews = parse_batch_result(text)
        except Exception as exc:
            failed.append((cid, f"parse_error: {exc}"))
            continue

        chunk_ids = manifest.get(cid, [])
        aliases = alias_map(chunk_ids)     # r1..rN -> real id (current scheme)
        expected = set(chunk_ids)          # raw ids (wave-1 records, fallback)
        for rv in reviews:
            emitted = rv.get("review_id", "")
            rid = aliases.get(emitted) or (emitted if emitted in expected else None)
            if rid is None:
                failed.append((cid, f"unknown review_id in output: {emitted!r}"))
                continue
            review_text = texts.get(rid, ("", ""))[1]
            aspects = normalize_aspects(rv.get("aspects", []), review_text)
            write_review_aspects(con, rid, aspects, label_model=record["model"])
            n_reviews += 1
            n_aspects += len(aspects)
            n_bad_evidence += sum(1 for a in aspects if not a["evidence_valid"])

    record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    record["n_reviews_written"] = n_reviews
    record["n_aspects_written"] = n_aspects
    record["n_invalid_evidence"] = n_bad_evidence
    record["n_failed_chunks"] = len(failed)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    rate = (n_bad_evidence / n_aspects * 100) if n_aspects else 0.0
    print(f"[{record_path.name}] wrote {n_reviews:,} reviews / "
          f"{n_aspects:,} aspect rows "
          f"(invalid evidence: {n_bad_evidence} = {rate:.1f}%)")
    for cid, why in failed:
        print(f"  ! {cid}: {why}")
    if failed:
        print("Unretrieved reviews stay unlabeled - re-run "
              "src/absa_label_submit.py to pick them up.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve ABSA batch results.")
    ap.add_argument("batch_files", nargs="*",
                    help="batch json files in batches/ (default: all pending absa)")
    ap.add_argument("--wait", action="store_true",
                    help=f"poll every {POLL_SECONDS}s until the batch ends")
    ap.add_argument("--redo", action="store_true",
                    help="also re-process batches already marked retrieved")
    args = ap.parse_args()

    if args.batch_files:
        paths = [BATCH_DIR / f if not Path(f).exists() else Path(f)
                 for f in args.batch_files]
    else:
        paths = []
        for p in sorted(BATCH_DIR.glob("absa_label_*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if args.redo or rec.get("retrieved_at") is None:
                paths.append(p)

    if not paths:
        print("No pending ABSA batches found in batches/.")
        return

    client = get_client()

    # Lazy write connection: opened only when a batch has ended and there is
    # something to write. Status checks never need (or take) the DB lock.
    _con: list[duckdb.DuckDBPyConnection] = []

    def get_con() -> duckdb.DuckDBPyConnection:
        if not _con:
            try:
                c = duckdb.connect(str(DB_PATH))
            except duckdb.IOException as exc:
                raise SystemExit(
                    f"Cannot open DB for writing: {exc}\n"
                    "Close any open notebook kernels / DuckDB connections "
                    "and re-run this script."
                )
            ensure_absa_tables(c)
            _con.append(c)
        return _con[0]

    try:
        for p in paths:
            retrieve_one(client, get_con, p, wait=args.wait)
    finally:
        if _con:
            _con[0].close()


if __name__ == "__main__":
    main()
