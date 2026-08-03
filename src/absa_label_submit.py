"""
absa_label_submit.py
====================
Submit review-level ABSA labeling requests to the Anthropic Message Batches API.

Chunks the sampled reviews (ABSA_SAMPLE, see absa_sample.py) into requests of
N_REVIEWS_PER_REQUEST reviews each. The batch id + chunk manifest is persisted
to batches/<name>.json so retrieval can happen later (results stay available
server-side for 29 days) - the PC can go offline in between.

Only reviews with no REVIEW_ASPECTS rows yet are submitted, so re-running
after a partial retrieve picks up exactly the missing ones.

Usage:
    # Submit all unlabeled sampled reviews
    uv run python src/absa_label_submit.py

    # Cap the number of reviews in this batch (e.g. a paid pilot)
    uv run python src/absa_label_submit.py --limit 5000

    # Inspect what would be sent without calling the API
    uv run python src/absa_label_submit.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import duckdb

from absa_label import (
    BATCH_DIR,
    DB_PATH,
    MAX_CHUNK_CHARS,
    MODEL,
    N_REVIEWS_PER_REQUEST,
    build_request,
    fetch_review_texts,
    get_client,
    pack_reviews,
    unlabeled_sample_ids,
)


def in_flight_ids() -> set[str]:
    """Review_ids already inside a submitted-but-not-yet-retrieved batch.

    Excluding them makes overlapping submissions impossible: you can fire
    wave 2 before retrieving wave 1 and never pay for a review twice.
    """
    ids: set[str] = set()
    for p in BATCH_DIR.glob("absa_label_*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("kind") == "absa_review_label" and rec.get("retrieved_at") is None:
            for chunk_ids in rec.get("chunks", {}).values():
                ids.update(chunk_ids)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit ABSA batch to Claude.")
    ap.add_argument("--limit", type=int, default=None,
                    help="max reviews to submit in this batch")
    ap.add_argument("--name", default=None,
                    help="batch file name (default: timestamp)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build requests and print stats; do not call the API")
    args = ap.parse_args()

    # Submit only READS the DB (the batch record is a JSON file), so a
    # read-only connection is enough - and it coexists with open notebook
    # kernels, unlike a write connection.
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        review_ids = unlabeled_sample_ids(con)  # sampled, not in REVIEW_ASPECTS
    except duckdb.CatalogException:
        print("ABSA tables missing - run src/absa_sample.py first.")
        con.close()
        return
    pending = in_flight_ids()                # sampled, sitting in a pending batch
    if pending:
        before = len(review_ids)
        review_ids = [r for r in review_ids if r not in pending]
        print(f"In-flight : {before - len(review_ids):,} reviews already in "
              f"pending batches - excluded (no double-labeling)")
    if args.limit:
        review_ids = review_ids[: args.limit]
    if not review_ids:
        print("Nothing to label - everything is labeled or in a pending batch. "
              "(Run src/absa_sample.py first if ABSA_SAMPLE is empty.)")
        con.close()
        return

    texts = fetch_review_texts(con, review_ids)
    con.close()

    all_reviews = [(rid, texts[rid][0], texts[rid][1])
                   for rid in review_ids if rid in texts]
    chunks = pack_reviews(all_reviews)

    requests, manifest = [], {}
    for i, reviews in enumerate(chunks):
        chunk_id = f"absa__c{i:05d}"
        requests.append(build_request(chunk_id, reviews))
        manifest[chunk_id] = [r[0] for r in reviews]

    n_reviews = sum(len(v) for v in manifest.values())
    sizes = [len(c) for c in chunks]
    print(f"Reviews : {n_reviews:,} in {len(requests)} requests "
          f"(<= {N_REVIEWS_PER_REQUEST} reviews & <= {MAX_CHUNK_CHARS} chars "
          f"per request; sizes min={min(sizes)} avg={sum(sizes)/len(sizes):.1f} "
          f"max={max(sizes)})")
    print(f"Model   : {MODEL} (Batches API, 50% off)")

    if args.dry_run:
        sample = requests[0]["params"]["messages"][0]["content"]
        print("\n--- first request user prompt (truncated) ---")
        print(sample[:1200])
        return

    client = get_client()
    batch = client.messages.batches.create(requests=requests)

    BATCH_DIR.mkdir(exist_ok=True)
    name = args.name or datetime.now().strftime("absa_label_%Y%m%d_%H%M%S")
    record = {
        "kind": "absa_review_label",
        "batch_id": batch.id,
        "model": MODEL,
        "n_requests": len(requests),
        "n_reviews": n_reviews,
        "chunks": manifest,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_at": None,
    }
    out = BATCH_DIR / f"{name}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Batch id : {batch.id}")
    print(f"Status   : {batch.processing_status}")
    print(f"Saved    : {out}")
    print("\nRetrieve later with:")
    print(f"  uv run python src/absa_label_retrieve.py {out.name}")


if __name__ == "__main__":
    main()
