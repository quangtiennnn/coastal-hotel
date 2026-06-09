"""
preprocess_to_duckdb.py
=======================
Batch-preprocess REVIEW_DATA.review_text with src/preprocessor.Preprocessor
(normalize + ViTokenizer/spaCy word-segmentation) and persist the results
into the REVIEW_TEXT_PROCESSED table in hotel_reviews.db.

Run this before embed_to_duckdb.py.

Usage
-----
    uv run python src/preprocess_to_duckdb.py
    uv run python src/preprocess_to_duckdb.py --db data/hotel_reviews.db --batch-size 10000

Behaviour
---------
- Idempotent: creates the table if missing, skips rows already present.
- Checkpoints every batch via NOT EXISTS — safe to interrupt and resume.
- ~6 min total for 251,328 rows on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

# Allow `from src.preprocessor import Preprocessor` when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import Preprocessor

DB_PATH = Path(__file__).parent.parent / "data" / "hotel_reviews.db"
TABLE = "REVIEW_TEXT_PROCESSED"


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            review_id      VARCHAR PRIMARY KEY,
            processed_text VARCHAR NOT NULL
        )
    """)


def preprocess_to_duckdb(db_path: Path = DB_PATH, batch_size: int = 10_000) -> None:
    con = duckdb.connect(str(db_path))
    _ensure_table(con)

    total_remaining = con.execute(f"""
        SELECT COUNT(*) FROM REVIEW_DATA r
        WHERE NOT EXISTS (SELECT 1 FROM {TABLE} p WHERE p.review_id = r.review_id)
          AND TRIM(COALESCE(r.review_text, '')) != ''
    """).fetchone()[0]

    if total_remaining == 0:
        print(f"[preprocess_to_duckdb] All rows already in {TABLE}. Nothing to do.")
        con.close()
        return

    print(f"[preprocess_to_duckdb] {total_remaining:,} rows to preprocess in batches of {batch_size:,}")

    preprocessor = Preprocessor()

    batch_num = 0
    total_done = 0

    while True:
        batch_df = con.execute(f"""
            SELECT r.review_id, r.review_text, r.language
            FROM REVIEW_DATA r
            WHERE NOT EXISTS (SELECT 1 FROM {TABLE} p WHERE p.review_id = r.review_id)
              AND TRIM(COALESCE(r.review_text, '')) != ''
            LIMIT ?
        """, [batch_size]).df()

        if batch_df.empty:
            break

        batch_num += 1
        n = len(batch_df)
        print(f"\n[batch {batch_num}] Processing {n:,} rows …")

        processed = preprocessor.process_texts(batch_df)

        con.executemany(
            f"INSERT OR IGNORE INTO {TABLE} (review_id, processed_text) VALUES (?, ?)",
            list(zip(batch_df["review_id"].tolist(), processed)),
        )

        total_done += n
        print(f"[batch {batch_num}] Done. Total so far: {total_done:,} / {total_remaining:,}")

    con.close()
    print(f"\n[preprocess_to_duckdb] Complete. {total_done:,} rows written to {TABLE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Preprocess REVIEW_DATA → {TABLE}")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--batch-size", type=int, default=10_000, dest="batch_size")
    args = parser.parse_args()
    preprocess_to_duckdb(db_path=args.db, batch_size=args.batch_size)
