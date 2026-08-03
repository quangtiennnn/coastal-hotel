"""
absa_gold.py
============
Assemble the gold-label sources for ABSA evaluation
(SENTENCES_LEVEL_PROPOSAL.md - "Ground truth").

Three free gold sources, written to two tables:

GOLD_REVIEW_ASPECTS  (in-domain, keyed to corpus review_ids)
  - gmap_tag        : reviewer 1-5 aspect stars (tag_room/service/location)
                      -> facility/service/amenity, star->sentiment
                      (1-2 neg, 3 neu, 4-5 pos). Only clean values {1..5}.
  - gmap_highlight  : Google highlight chips -> experience (POSITIVE ONLY)
                      + Cong nghe cao -> facility. "Phu hop voi tre em" dropped.

GOLD_TRIPADVISOR     (out-of-domain VALIDATION benchmark, en only)
  - span-level human annotations from data/raw/Annoted_TripAdvisor_EN.json
    (Label Studio format). Aspect labels map 1:1 onto the 5 macro aspects;
    Branding -> other. Full sentiment incl. negatives. Validation only -
    never for training.

Usage:
    uv run python src/absa_gold.py            # build all gold tables
    uv run python src/absa_gold.py --stats    # only print what exists
"""

from __future__ import annotations

import argparse
import json

import duckdb

from absa_label import DB_PATH, ROOT

TRIPADVISOR_JSON = ROOT / "data" / "raw" / "Annoted_TripAdvisor_EN.json"
# Second out-of-domain benchmark. Despite the name, texts are ENGLISH
# ("VN" = Vietnam hotels): 7,000/7,008 en. The TripAdvisor_EN/VN files in the
# same folder are byte-identical duplicates of Annoted_TripAdvisor_EN.json.
BOOKING_JSON = ROOT / "data" / "raw" / "Annotated Data" / "BOOKING_VN.json"

# --- Google Maps reviewer star tags -> macro aspect -----------------------

TAG_TO_KEY = {
    "tag_room": "facility",
    "tag_service": "service",
    "tag_location": "amenity",
    # tag_food_drink: free text, unusable as star gold (see proposal)
}
STAR_TO_SENTIMENT = {
    "1": "negative", "2": "negative",
    "3": "neutral",
    "4": "positive", "5": "positive",
}

# --- Google Maps highlight chips (vi) -> (key_aspect, sub_aspect) ----------
# Highlights are POSITIVE-ONLY by construction.

HIGHLIGHT_MAP = {
    "Giá tốt": ("experience", "price_value"),
    "Cảnh đẹp": ("experience", "coastal_view"),
    "Yên tĩnh": ("experience", "noise_quietness"),
    "Sang trọng": ("experience", "overall_atmosphere"),
    "Lãng mạn": ("experience", "overall_atmosphere"),
    "Thoải mái": ("experience", "comfort_relaxation"),
    "Công nghệ cao": ("facility", "technical_equipment"),
    # "Phù hợp với trẻ em": ambiguous -> dropped (see proposal)
}

# --- TripAdvisor annotation labels -> macro aspect --------------------------

TA_ASPECT_MAP = {
    "Facility": "facility",
    "Amenity": "amenity",
    "Service": "service",
    "Experience": "experience",
    "Loyalty": "loyalty",
    "Branding": "other",  # no home in the 5-aspect scheme
}
TA_SENTIMENT_MAP = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}


def ensure_gold_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS GOLD_REVIEW_ASPECTS (
            review_id   VARCHAR NOT NULL,
            key_aspect  VARCHAR NOT NULL,
            sub_aspect  VARCHAR,            -- only highlights carry one
            sentiment   VARCHAR NOT NULL,
            gold_source VARCHAR NOT NULL,   -- gmap_tag / gmap_highlight
            raw_value   VARCHAR,            -- original star / chip
            PRIMARY KEY (review_id, key_aspect, gold_source, raw_value)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS GOLD_TRIPADVISOR (
            doc_id      VARCHAR NOT NULL,   -- ta_<item id>
            span_id     VARCHAR NOT NULL,   -- Label Studio result id
            key_aspect  VARCHAR NOT NULL,   -- mapped; Branding -> other
            sentiment   VARCHAR,            -- may be missing on odd spans
            evidence    VARCHAR,            -- annotated text span
            span_start  INTEGER,
            span_end    INTEGER,
            review_text VARCHAR,            -- full document text
            PRIMARY KEY (doc_id, span_id, key_aspect)
        )
    """)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_gmap_tag_gold(con: duckdb.DuckDBPyConnection) -> int:
    con.execute("DELETE FROM GOLD_REVIEW_ASPECTS WHERE gold_source = 'gmap_tag'")
    n = 0
    for tag_col, key_aspect in TAG_TO_KEY.items():
        con.execute(f"""
            INSERT INTO GOLD_REVIEW_ASPECTS
              (review_id, key_aspect, sub_aspect, sentiment, gold_source, raw_value)
            SELECT review_id, '{key_aspect}', NULL,
                   CASE WHEN {tag_col} IN ('1','2') THEN 'negative'
                        WHEN {tag_col} = '3'        THEN 'neutral'
                        ELSE 'positive' END,
                   'gmap_tag', {tag_col}
            FROM GOOGLEMAPS_REVIEW
            WHERE {tag_col} IN ('1','2','3','4','5')
        """)
        n += con.execute(f"""
            SELECT count(*) FROM GOOGLEMAPS_REVIEW
            WHERE {tag_col} IN ('1','2','3','4','5')
        """).fetchone()[0]
    return n


def build_gmap_highlight_gold(con: duckdb.DuckDBPyConnection) -> int:
    con.execute("DELETE FROM GOLD_REVIEW_ASPECTS WHERE gold_source = 'gmap_highlight'")
    rows = con.execute("""
        SELECT review_id, tag_highlights
        FROM GOOGLEMAPS_REVIEW
        WHERE tag_highlights IS NOT NULL
    """).fetchall()

    inserts = []
    for review_id, raw in rows:
        for chip in raw.split(","):
            chip = chip.strip().rstrip("…").strip()
            mapped = HIGHLIGHT_MAP.get(chip)
            if mapped is None:
                continue  # truncated / unknown / dropped chips
            key_aspect, sub_aspect = mapped
            inserts.append((review_id, key_aspect, sub_aspect,
                            "positive", "gmap_highlight", chip))
    if inserts:
        con.executemany("""
            INSERT OR REPLACE INTO GOLD_REVIEW_ASPECTS
              (review_id, key_aspect, sub_aspect, sentiment, gold_source, raw_value)
            VALUES (?, ?, ?, ?, ?, ?)
        """, inserts)
    return len(inserts)


def parse_labelstudio(path, prefix: str) -> tuple[list[tuple], int]:
    """Parse a Label Studio export: pair aspect + sentiment spans by result id.

    Returns (insert rows for a GOLD_* span table, n annotated docs).
    """
    items = json.loads(path.read_text(encoding="utf-8"))

    inserts, n_docs = [], 0
    for item in items:
        doc_id = f"{prefix}_{item['id']}"
        text = (item.get("data") or {}).get("text", "") or ""
        spans: dict[str, dict] = {}
        for ann in item.get("annotations", []):
            for res in ann.get("result", []):
                rid = res.get("id")
                val = res.get("value") or {}
                labels = val.get("labels") or []
                if rid is None or not labels:
                    continue
                span = spans.setdefault(rid, {
                    "start": val.get("start"), "end": val.get("end"),
                    "text": val.get("text", ""),
                    "aspect": None, "sentiment": None,
                })
                label = labels[0]
                if res.get("from_name") == "entities":
                    span["aspect"] = TA_ASPECT_MAP.get(label)
                elif res.get("from_name") == "entity_sentiment":
                    span["sentiment"] = TA_SENTIMENT_MAP.get(label)
        wrote = False
        for rid, s in spans.items():
            if s["aspect"] is None:
                continue  # sentiment-only orphan span
            inserts.append((doc_id, rid, s["aspect"], s["sentiment"],
                            s["text"], s["start"], s["end"], text))
            wrote = True
        if wrote:
            n_docs += 1
    return inserts, n_docs


def build_span_gold(con: duckdb.DuckDBPyConnection, table: str,
                    path, prefix: str) -> tuple[int, int]:
    """Load one Label Studio benchmark into its GOLD_* span table."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            doc_id      VARCHAR NOT NULL,
            span_id     VARCHAR NOT NULL,
            key_aspect  VARCHAR NOT NULL,
            sentiment   VARCHAR,
            evidence    VARCHAR,
            span_start  INTEGER,
            span_end    INTEGER,
            review_text VARCHAR,
            PRIMARY KEY (doc_id, span_id, key_aspect)
        )
    """)
    inserts, n_docs = parse_labelstudio(path, prefix)
    con.execute(f"DELETE FROM {table}")
    con.executemany(f"""
        INSERT OR REPLACE INTO {table}
          (doc_id, span_id, key_aspect, sentiment, evidence,
           span_start, span_end, review_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, inserts)
    return n_docs, len(inserts)


def print_stats(con: duckdb.DuckDBPyConnection) -> None:
    print("\n=== GOLD_REVIEW_ASPECTS ===")
    for r in con.execute("""
        SELECT gold_source, key_aspect, sentiment, count(*)
        FROM GOLD_REVIEW_ASPECTS GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """).fetchall():
        print(f"  {r[0]:15} {r[1]:11} {r[2]:9} {r[3]:7,}")
    for table in ("GOLD_TRIPADVISOR", "GOLD_BOOKING"):
        print(f"\n=== {table} ===")
        for r in con.execute(f"""
            SELECT key_aspect, sentiment, count(*)
            FROM {table} GROUP BY 1, 2 ORDER BY 1, 2
        """).fetchall():
            print(f"  {r[0]:11} {str(r[1]):9} {r[2]:7,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ABSA gold tables.")
    ap.add_argument("--stats", action="store_true", help="print stats only")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))
    ensure_gold_tables(con)
    if not args.stats:
        n_tag = build_gmap_tag_gold(con)
        print(f"gmap_tag gold      : {n_tag:,} rows")
        n_hl = build_gmap_highlight_gold(con)
        print(f"gmap_highlight gold: {n_hl:,} rows (positive-only)")
        n_docs, n_spans = build_span_gold(con, "GOLD_TRIPADVISOR",
                                          TRIPADVISOR_JSON, "ta")
        print(f"tripadvisor gold   : {n_docs:,} docs / {n_spans:,} spans "
              f"(VALIDATION ONLY - out-of-domain, en)")
        n_docs, n_spans = build_span_gold(con, "GOLD_BOOKING",
                                          BOOKING_JSON, "bk")
        print(f"booking gold       : {n_docs:,} docs / {n_spans:,} spans "
              f"(VALIDATION ONLY - out-of-domain, en despite the VN name)")
    print_stats(con)
    con.close()


if __name__ == "__main__":
    main()
