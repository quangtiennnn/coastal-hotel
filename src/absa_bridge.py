"""
absa_bridge.py
==============
Review -> sentence bridge for Model B training data
(SENTENCES_LEVEL_PROPOSAL.md - "Unit of analysis").

The LLM labels whole reviews with evidence spans (verbatim substrings).
Model B trains on segmented sentences. This script converts one into the
other: segment each labeled review into sentences, locate every evidence
span in the review text, and give each sentence the (key_aspect, sub_aspect,
sentiment) of the span(s) overlapping it.

Segmentation: spaCy rule-based sentencizer - spacy.blank("en") for English,
spacy.blank("vi") for Vietnamese (uses the installed pyvi tokenizer).
NOTE: the proposal names VnCoreNLP/underthesea for vi; underthesea is not
installed, so the sentencizer stands in until it is (swap in segment_review()).

Output: SENTENCE_LABELS - one row per (sentence, overlapping aspect span).
Sentences with no overlapping span get a single key_aspect='none' row
(explicit negatives, generated programmatically per the proposal's sparse-
output rule - never asked of the model).

Usage:
    uv run python src/absa_bridge.py            # bridge all labeled reviews
    uv run python src/absa_bridge.py --limit 50 # smoke run
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import duckdb

from absa_label import DB_PATH, find_evidence

MIN_SENT_CHARS = 3


@lru_cache(maxsize=2)
def _nlp(lang: str):
    import spacy

    lang = "vi" if lang == "vi" else "en"
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    return nlp


def segment_review(text: str, lang: str) -> list[tuple[int, int, str]]:
    """[(start, end, sentence), ...] with character offsets into `text`."""
    doc = _nlp(lang)(text)
    return [
        (s.start_char, s.end_char, s.text.strip())
        for s in doc.sents
        if len(s.text.strip()) >= MIN_SENT_CHARS
    ]


def locate_spans(text: str, aspects: list[dict]) -> list[dict]:
    """Attach (start, end) to each aspect via substring search.

    Repeated identical evidence strings advance the search cursor so two
    'phong sach' spans land on two different occurrences.
    """
    cursor_by_ev: dict[str, int] = {}
    located = []
    for a in aspects:
        ev = a["evidence"] or ""
        if not ev:
            continue
        span = find_evidence(text, ev, cursor_by_ev.get(ev, 0))
        if span is None:
            span = find_evidence(text, ev)  # fall back to first occurrence
        if span is None:
            continue  # hallucinated evidence - skip (evidence_valid False)
        start, end = span
        cursor_by_ev[ev] = start + 1
        located.append({**a, "start": start, "end": end})
    return located


def ensure_sentence_table(con: duckdb.DuckDBPyConnection) -> None:
    # No PRIMARY KEY: 'none' rows carry NULL sub_aspect/sentiment (DuckDB PK
    # columns are implicitly NOT NULL). Idempotency comes from the
    # DELETE-per-review before insert; duplicates are removed in Python.
    con.execute("""
        CREATE TABLE IF NOT EXISTS SENTENCE_LABELS (
            review_id   VARCHAR NOT NULL,
            sent_idx    INTEGER NOT NULL,
            sentence    VARCHAR NOT NULL,
            key_aspect  VARCHAR NOT NULL,   -- 'none' = explicit negative
            sub_aspect  VARCHAR,
            sentiment   VARCHAR,            -- NULL for 'none' rows
            language    VARCHAR,
            split       VARCHAR             -- from ABSA_SAMPLE (may be NULL)
        )
    """)


def bridge_review(
    review_id: str, text: str, lang: str, split: str | None,
    aspects: list[dict],
) -> list[tuple]:
    sentences = segment_review(text, lang)
    spans = locate_spans(text, aspects)

    rows: list[tuple] = []
    seen: set[tuple] = set()
    for idx, (s_start, s_end, sent) in enumerate(sentences):
        hits = [sp for sp in spans if sp["start"] < s_end and sp["end"] > s_start]
        if hits:
            for sp in hits:
                row = (review_id, idx, sent, sp["key_aspect"],
                       sp["sub_aspect"], sp["sentiment"], lang, split)
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
        else:
            rows.append((review_id, idx, sent, "none", None, None, lang, split))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Review->sentence bridge.")
    ap.add_argument("--limit", type=int, default=None,
                    help="max reviews to bridge (smoke run)")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))
    ensure_sentence_table(con)

    labeled = con.execute("""
        SELECT DISTINCT ra.review_id, r.review_text, r.language, s.split
        FROM REVIEW_ASPECTS ra
        JOIN REVIEW_DATA r USING (review_id)
        LEFT JOIN ABSA_SAMPLE s USING (review_id)
        WHERE r.review_text IS NOT NULL
        ORDER BY ra.review_id
    """).fetchall()
    if args.limit:
        labeled = labeled[: args.limit]
    print(f"Bridging {len(labeled):,} labeled reviews ...")

    n_rows = 0
    for review_id, text, lang, split in labeled:
        aspects = [
            {"key_aspect": k, "sub_aspect": sub, "sentiment": sen, "evidence": ev}
            for k, sub, sen, ev in con.execute("""
                SELECT key_aspect, sub_aspect, sentiment, evidence
                FROM REVIEW_ASPECTS
                WHERE review_id = ? AND evidence_valid
                ORDER BY aspect_rank
            """, [review_id]).fetchall()
        ]
        rows = bridge_review(review_id, text, lang or "en", split, aspects)
        con.execute("DELETE FROM SENTENCE_LABELS WHERE review_id = ?", [review_id])
        if rows:
            con.executemany("""
                INSERT INTO SENTENCE_LABELS
                  (review_id, sent_idx, sentence, key_aspect, sub_aspect,
                   sentiment, language, split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
        n_rows += len(rows)

    stats = con.execute("""
        SELECT key_aspect, count(*) FROM SENTENCE_LABELS
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    print(f"Wrote {n_rows:,} sentence rows:")
    for k, n in stats:
        print(f"  {k:11} {n:7,}")
    con.close()


if __name__ == "__main__":
    main()
