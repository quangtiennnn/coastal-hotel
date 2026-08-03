"""
absa_sample.py
==============
Topic-diversity sampling for the review-level ABSA labeling set
(SENTENCES_LEVEL_PROPOSAL.md - "Sampling strategy").

BERTopic's only role here is review SELECTION - it does not define labels.

Method (coverage-first):
1. Frame = the 6 per-language coast-band runs in REVIEW_TOPICS
   (coast_band_{A,B,C}_{en,vi}) - together they cover the full 174k corpus,
   and run_id encodes language x coast band, so per-topic quotas are already
   stratified by both.
2. Allocate the budget ACROSS (run, topic) cells with a square-root quota
   (sqrt(n_topic) weighting) so huge generic topics are capped and the long
   tail survives; every topic gets a floor of MIN_PER_TOPIC.
3. Rare-aspect boost: topics whose primary TOPIC_ASPECTS label is a rare
   macro aspect (loyalty) or negative sentiment get a quota multiplier.
4. Within a topic, reviews are ordered by md5(review_id) - deterministic,
   reproducible sampling; sources (agoda/googlemaps) mix naturally.
5. 80/10/10 train/val/test split, again by md5 hash.

Writes the result to ABSA_SAMPLE (replacing any previous sample).

Usage:
    uv run python src/absa_sample.py                 # default 15,000
    uv run python src/absa_sample.py --n 5000        # pilot
    uv run python src/absa_sample.py --dry-run       # print allocation only
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import duckdb

from absa_label import DB_PATH, ensure_absa_tables

RUNS = [
    "coast_band_A_en", "coast_band_A_vi",
    "coast_band_B_en", "coast_band_B_vi",
    "coast_band_C_en", "coast_band_C_vi",
]
MIN_PER_TOPIC = 10       # floor so small topics are never starved
MIN_REVIEW_CHARS = 30    # skip near-empty reviews
OUTLIER_TOPIC = -1       # BERTopic outlier bucket: included but never boosted

# Rare-cell oversampling (proposal step 4). Multipliers on the sqrt quota.
BOOST_ASPECT = {"loyalty": 3.0}      # loyalty is the rarest macro aspect
BOOST_NEGATIVE = 1.5                 # negative-sentiment topics


def topic_boosts(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, int], float]:
    """(run_id, topic_id) -> quota multiplier from the topic's primary label."""
    rows = con.execute("""
        SELECT run_id, topic_id, key_aspect, sentiment
        FROM TOPIC_ASPECTS
        WHERE is_primary
    """).fetchall()
    boosts: dict[tuple[str, int], float] = {}
    for run_id, topic_id, key_aspect, sentiment in rows:
        m = BOOST_ASPECT.get(key_aspect, 1.0)
        if sentiment == "negative":
            m *= BOOST_NEGATIVE
        if m != 1.0:
            boosts[(run_id, int(topic_id))] = m
    return boosts


def allocate(
    sizes: dict[tuple[str, int], int],
    boosts: dict[tuple[str, int], float],
    total: int,
) -> dict[tuple[str, int], int]:
    """Square-root allocation with floor + cap, scaled to `total`.

    Cells capped at their topic size return their unused budget, which is
    redistributed across the uncapped cells until the target is met (or
    every cell is saturated).
    """
    weights = {
        cell: math.sqrt(n) * boosts.get(cell, 1.0)
        for cell, n in sizes.items()
    }
    quotas = {cell: min(MIN_PER_TOPIC, n) for cell, n in sizes.items()}

    # Iteratively hand the remaining budget to unsaturated cells by weight.
    for _ in range(50):
        remaining = total - sum(quotas.values())
        if remaining <= 0:
            break
        open_cells = [c for c in sizes if quotas[c] < sizes[c]]
        if not open_cells:
            break
        wsum = sum(weights[c] for c in open_cells)
        progressed = False
        for c in open_cells:
            add = min(int(remaining * weights[c] / wsum), sizes[c] - quotas[c])
            if add > 0:
                quotas[c] += add
                progressed = True
        if not progressed:  # rounding stall: give 1-by-1 to heaviest open cells
            for c in sorted(open_cells, key=lambda c: weights[c], reverse=True):
                if sum(quotas.values()) >= total:
                    break
                quotas[c] += 1
            break

    # The floor can push the sum above `total`; shave from the largest quotas.
    excess = sum(quotas.values()) - total
    if excess > 0:
        for cell in sorted(quotas, key=quotas.get, reverse=True):
            if excess <= 0:
                break
            reducible = quotas[cell] - min(MIN_PER_TOPIC, sizes[cell])
            take = min(reducible, excess)
            quotas[cell] -= take
            excess -= take
    return quotas


def main() -> None:
    ap = argparse.ArgumentParser(description="Topic-diversity ABSA sampling.")
    ap.add_argument("--n", type=int, default=15000, help="total sample size")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the allocation without writing ABSA_SAMPLE")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))
    ensure_absa_tables(con)

    # Candidate frame: one row per review with its topic cell
    placeholders = ", ".join("?" for _ in RUNS)
    frame = con.execute(f"""
        SELECT rt.run_id, rt.topic_id, rt.review_id, r.source, r.language
        FROM REVIEW_TOPICS rt
        JOIN REVIEW_DATA r USING (review_id)
        WHERE rt.run_id IN ({placeholders})
          AND r.review_text IS NOT NULL
          AND length(r.review_text) >= {MIN_REVIEW_CHARS}
        ORDER BY md5(rt.review_id)
    """, RUNS).fetchall()

    cells: dict[tuple[str, int], list[tuple[str, str, str]]] = defaultdict(list)
    for run_id, topic_id, review_id, source, language in frame:
        cells[(run_id, int(topic_id))].append((review_id, source, language))

    sizes = {cell: len(rows) for cell, rows in cells.items()}
    boosts = topic_boosts(con)
    boosts = {c: m for c, m in boosts.items()
              if c in sizes and c[1] != OUTLIER_TOPIC}
    quotas = allocate(sizes, boosts, args.n)

    print(f"Frame: {sum(sizes.values()):,} reviews in {len(sizes)} (run, topic) cells")
    print(f"Boosted cells: {len(boosts)}  |  Target: {args.n:,}  |  "
          f"Allocated: {sum(quotas.values()):,}")
    for run_id in RUNS:
        run_cells = [c for c in quotas if c[0] == run_id]
        print(f"  {run_id}: {len(run_cells)} topics, "
              f"{sum(quotas[c] for c in run_cells):,} reviews")

    if args.dry_run:
        con.close()
        return

    picked: list[tuple[str, str, int, str, str]] = []
    for cell, quota in quotas.items():
        run_id, topic_id = cell
        for review_id, source, language in cells[cell][:quota]:
            picked.append((review_id, run_id, topic_id, source, language))

    # Deterministic 80/10/10 split on the review hash
    def split_of(review_id: str) -> str:
        h = con.execute("SELECT md5(?)", [review_id]).fetchone()[0]
        bucket = int(h[:8], 16) % 10
        return "train" if bucket < 8 else ("val" if bucket == 8 else "test")

    con.execute("DELETE FROM ABSA_SAMPLE")
    con.executemany("""
        INSERT OR REPLACE INTO ABSA_SAMPLE
          (review_id, run_id, topic_id, source, language, split, sampled_at)
        VALUES (?, ?, ?, ?, ?, ?, now())
    """, [(rid, run, tid, src, lang, split_of(rid))
          for rid, run, tid, src, lang in picked])

    stats = con.execute("""
        SELECT split, language, source, count(*)
        FROM ABSA_SAMPLE GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """).fetchall()
    print(f"\nWrote {len(picked):,} rows to ABSA_SAMPLE:")
    for split, lang, src, n in stats:
        print(f"  {split:5} {lang} {src:11} {n:6,}")
    con.close()


if __name__ == "__main__":
    main()
