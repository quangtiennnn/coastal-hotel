"""
embed_to_duckdb.py
==================
Batch-encode processed review text with paraphrase-multilingual-mpnet-base-v2
and persist the 768-dim vectors into the REVIEW_EMBEDDINGS table in hotel_reviews.db.

Reads processed_text from REVIEW_TEXT_PROCESSED — run preprocess_to_duckdb.py first.

Usage
-----
    uv run python src/embed_to_duckdb.py
    uv run python src/embed_to_duckdb.py --db data/hotel_reviews.db --batch-size 10000

Behaviour
---------
- Idempotent: creates the table if missing, skips rows already present.
- Checkpoints every batch via NOT EXISTS — safe to interrupt and resume.
- ~38 min total for 251,328 rows on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).parent.parent / "data" / "hotel_reviews.db"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
VECTOR_DIM = 768
SOURCE_TABLE = "REVIEW_TEXT_PROCESSED"
TARGET_TABLE = "REVIEW_EMBEDDINGS"


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    # Drop old schema if it has the wrong columns (processed_text moved to REVIEW_TEXT_PROCESSED)
    existing_cols = {
        row[0] for row in con.execute(f"DESCRIBE {TARGET_TABLE}").fetchall()
    } if _table_exists(con, TARGET_TABLE) else set()

    if "processed_text" in existing_cols:
        row_count = con.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}").fetchone()[0]
        if row_count == 0:
            con.execute(f"DROP TABLE {TARGET_TABLE}")
            print(f"[schema] Dropped old {TARGET_TABLE} (had wrong columns, was empty).")
        else:
            raise RuntimeError(
                f"{TARGET_TABLE} has the old schema (includes processed_text) and {row_count:,} rows. "
                "Migrate manually or clear the table before running."
            )

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            review_id VARCHAR PRIMARY KEY,
            embedding FLOAT[{VECTOR_DIM}] NOT NULL
        )
    """)


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return name in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def embed_to_duckdb(db_path: Path = DB_PATH, batch_size: int = 10_000) -> None:
    con = duckdb.connect(str(db_path))

    if not _table_exists(con, SOURCE_TABLE):
        raise RuntimeError(
            f"{SOURCE_TABLE} does not exist. "
            "Run `uv run python src/preprocess_to_duckdb.py` first."
        )

    source_count = con.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
    if source_count == 0:
        raise RuntimeError(
            f"{SOURCE_TABLE} is empty. "
            "Run `uv run python src/preprocess_to_duckdb.py` first."
        )

    _ensure_table(con)

    total_remaining = con.execute(f"""
        SELECT COUNT(*) FROM {SOURCE_TABLE} p
        WHERE NOT EXISTS (SELECT 1 FROM {TARGET_TABLE} e WHERE e.review_id = p.review_id)
    """).fetchone()[0]

    if total_remaining == 0:
        print(f"[embed_to_duckdb] All rows already in {TARGET_TABLE}. Nothing to do.")
        con.close()
        return

    print(f"[embed_to_duckdb] {total_remaining:,} rows to embed in batches of {batch_size:,}")
    print(f"[embed_to_duckdb] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    batch_num = 0
    total_done = 0

    while True:
        batch_df = con.execute(f"""
            SELECT p.review_id, p.processed_text
            FROM {SOURCE_TABLE} p
            WHERE NOT EXISTS (SELECT 1 FROM {TARGET_TABLE} e WHERE e.review_id = p.review_id)
            LIMIT ?
        """, [batch_size]).df()

        if batch_df.empty:
            break

        batch_num += 1
        n = len(batch_df)
        print(f"\n[batch {batch_num}] Encoding {n:,} rows …")

        embeddings: np.ndarray = model.encode(
            batch_df["processed_text"].tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        con.executemany(
            f"INSERT OR IGNORE INTO {TARGET_TABLE} (review_id, embedding) VALUES (?, ?)",
            [(rid, emb.tolist()) for rid, emb in zip(batch_df["review_id"].tolist(), embeddings)],
        )

        total_done += n
        print(f"[batch {batch_num}] Done. Total so far: {total_done:,} / {total_remaining:,}")

    con.close()
    print(f"\n[embed_to_duckdb] Complete. {total_done:,} rows written to {TARGET_TABLE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Embed {SOURCE_TABLE} → {TARGET_TABLE}")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--batch-size", type=int, default=10_000, dest="batch_size")
    args = parser.parse_args()
    embed_to_duckdb(db_path=args.db, batch_size=args.batch_size)
