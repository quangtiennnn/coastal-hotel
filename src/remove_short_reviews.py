"""
remove_short_reviews.py
=======================
Remove reviews with fewer than 3 words from hotel_reviews.db and wipe
all downstream topic/label tables so they can be cleanly re-generated.

Affected tables (in deletion order):
  1. TOPIC_ASPECTS          -- silver labels (re-run llm_label_submit + retrieve)
  2. TOPIC_LABELS           -- topic keywords (re-run notebook 21)
  3. REVIEW_TOPICS          -- per-review topic assignments (re-run notebook 21)
  4. REVIEW_EMBEDDINGS      -- 768-dim vectors for short reviews
  5. REVIEW_TEXT_PROCESSED  -- preprocessed text for short reviews
  6. AGODA_REVIEW           -- source rows with < 3 words
  7. GOOGLEMAPS_REVIEW      -- source rows with < 3 words

Usage
-----
    uv run python src/remove_short_reviews.py
    uv run python src/remove_short_reviews.py --db data/hotel_reviews.db --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).parent.parent / "data" / "hotel_reviews.db"
MIN_WORDS = 5


def word_count_expr(col: str) -> str:
    return f"array_length(str_split(trim({col}), chr(32)))"


def report(con: duckdb.DuckDBPyConnection) -> None:
    short_agoda = con.execute(
        f"SELECT COUNT(*) FROM AGODA_REVIEW WHERE {word_count_expr('review_text')} < {MIN_WORDS}"
    ).fetchone()[0]
    short_gmap = con.execute(
        f"SELECT COUNT(*) FROM GOOGLEMAPS_REVIEW WHERE {word_count_expr('review_text')} < {MIN_WORDS}"
    ).fetchone()[0]
    total_review_data = con.execute("SELECT COUNT(*) FROM REVIEW_DATA").fetchone()[0]
    total_processed   = con.execute("SELECT COUNT(*) FROM REVIEW_TEXT_PROCESSED").fetchone()[0]
    total_embeddings  = con.execute("SELECT COUNT(*) FROM REVIEW_EMBEDDINGS").fetchone()[0]
    total_topics      = con.execute("SELECT COUNT(*) FROM REVIEW_TOPICS").fetchone()[0]
    total_labels      = con.execute("SELECT COUNT(*) FROM TOPIC_LABELS").fetchone()[0]
    total_aspects     = con.execute("SELECT COUNT(*) FROM TOPIC_ASPECTS").fetchone()[0]

    print("=" * 55)
    print("Current DB state")
    print("=" * 55)
    print(f"  AGODA_REVIEW short rows to delete     : {short_agoda:,}")
    print(f"  GOOGLEMAPS_REVIEW short rows to delete: {short_gmap:,}")
    print(f"  REVIEW_DATA (view) total              : {total_review_data:,}")
    print(f"  REVIEW_TEXT_PROCESSED total           : {total_processed:,}")
    print(f"  REVIEW_EMBEDDINGS total               : {total_embeddings:,}")
    print(f"  REVIEW_TOPICS total                   : {total_topics:,}")
    print(f"  TOPIC_LABELS total                    : {total_labels:,}")
    print(f"  TOPIC_ASPECTS (silver labels)         : {total_aspects:,}")
    print("=" * 55)


def run(db_path: Path, dry_run: bool) -> None:
    con = duckdb.connect(str(db_path))

    print(f"\n[remove_short_reviews] DB: {db_path}")
    print(f"[remove_short_reviews] MIN_WORDS = {MIN_WORDS}")
    print(f"[remove_short_reviews] dry_run   = {dry_run}\n")

    report(con)

    if dry_run:
        print("\n[dry-run] No changes made.")
        con.close()
        return

    print("\nStarting deletion ...\n")

    # 1. Collect short review_ids (need for keyed deletes)
    print("[1/7] Collecting short review IDs from AGODA_REVIEW ...")
    short_agoda_ids = con.execute(
        f"SELECT review_id FROM AGODA_REVIEW WHERE {word_count_expr('review_text')} < {MIN_WORDS}"
    ).df()["review_id"].tolist()
    print(f"      {len(short_agoda_ids):,} IDs collected.")

    print("[2/7] Collecting short review IDs from GOOGLEMAPS_REVIEW ...")
    short_gmap_ids = con.execute(
        f"SELECT review_id FROM GOOGLEMAPS_REVIEW WHERE {word_count_expr('review_text')} < {MIN_WORDS}"
    ).df()["review_id"].tolist()
    print(f"      {len(short_gmap_ids):,} IDs collected.")

    all_short_ids = short_agoda_ids + [str(i) for i in short_gmap_ids]

    # 2. Wipe downstream tables first (foreign-key order)
    print("[3/7] Wiping TOPIC_ASPECTS ...")
    con.execute("DELETE FROM TOPIC_ASPECTS")
    print("      Done.")

    print("[4/7] Wiping TOPIC_LABELS ...")
    con.execute("DELETE FROM TOPIC_LABELS")
    print("      Done.")

    print("[5/7] Wiping REVIEW_TOPICS ...")
    con.execute("DELETE FROM REVIEW_TOPICS")
    print("      Done.")

    # 3. Remove embeddings + processed text for short reviews
    print("[6/7] Removing short reviews from REVIEW_EMBEDDINGS and REVIEW_TEXT_PROCESSED ...")
    batch_size = 5_000
    for i in range(0, len(all_short_ids), batch_size):
        batch = all_short_ids[i : i + batch_size]
        placeholders = ", ".join("?" * len(batch))
        con.execute(f"DELETE FROM REVIEW_EMBEDDINGS WHERE review_id IN ({placeholders})", batch)
        con.execute(f"DELETE FROM REVIEW_TEXT_PROCESSED WHERE review_id IN ({placeholders})", batch)
    print(f"      Removed {len(all_short_ids):,} rows from each.")

    # 4. Remove from source tables
    print("[7/7] Removing short reviews from AGODA_REVIEW and GOOGLEMAPS_REVIEW ...")
    deleted_agoda = con.execute(
        f"DELETE FROM AGODA_REVIEW WHERE {word_count_expr('review_text')} < {MIN_WORDS}"
    ).fetchone()
    deleted_gmap = con.execute(
        f"DELETE FROM GOOGLEMAPS_REVIEW WHERE {word_count_expr('review_text')} < {MIN_WORDS}"
    ).fetchone()
    print(f"      AGODA_REVIEW     : rows deleted.")
    print(f"      GOOGLEMAPS_REVIEW: rows deleted.")

    print("\n[remove_short_reviews] All deletions complete. Final DB state:\n")
    report(con)

    con.close()
    print("\n[remove_short_reviews] Done.")
    print("\nNext steps:")
    print("  1. Re-run notebook 21 (21_refit_all_runs.ipynb) to regenerate topic models")
    print("  2. uv run python src/llm_label_submit.py   -- resubmit silver labels to Claude")
    print("  3. uv run python src/llm_label_retrieve.py -- retrieve and store results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove short reviews from hotel_reviews.db")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without deleting")
    args = parser.parse_args()
    run(db_path=args.db, dry_run=args.dry_run)
