"""
llm_label.py
============
Claude-assisted silver-labeling of BERTopic topics (PROPOSAL_COASTAL_TOPIC_ANALYSIS_V2).

Maps each topic in TOPIC_LABELS onto a fixed 5-aspect taxonomy
(facility / amenity / service / experience / loyalty) with 1-3 aspects per
topic, each carrying its own sentiment, weight, and free-form sub_aspects.

Model:     claude-sonnet-4-6 via the Anthropic Message Batches API (50% off)
Output:    enforced JSON via output_config.format (json_schema)
Storage:   TOPIC_LABELS (topic-level columns added) + TOPIC_ASPECTS (new, one
           row per topic-aspect pair) in data/hotel_reviews.db

Used by:   src/llm_label_submit.py, src/llm_label_retrieve.py,
           notebooks/17_silver_label_test.ipynb
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import duckdb
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "hotel_reviews.db"
BATCH_DIR = ROOT / "batches"

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
N_SAMPLE_DOCS = 5
DOC_MAX_CHARS = 480  # ~120 tokens per excerpt
N_TOP_WORDS = 10

load_dotenv(ROOT / ".env")


def get_client():
    """Anthropic client. Reads CLAUDE_API_KEY (preferred) or ANTHROPIC_API_KEY from .env."""
    import anthropic

    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Add CLAUDE_API_KEY=sk-ant-... to the .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


# ===========================================================================
# Taxonomy + prompt
# ===========================================================================

KEY_ASPECTS: dict[str, str] = {
    "facility": (
        "Facility, room, furnishings, bathroom, charging, reception area, "
        "restaurant facilities, public equipment, gym, pool, elevator, bed, "
        "mattress, water, electricity, configuration, home appliances, article, "
        "supplies, utensils, decoration, air conditioner, infrastructure, "
        "environment, sanitation, sound insulation, ventilation, natural lighting, "
        "landscape, scenery, new, old, complete, antiquated, shabby, wet, dark, "
        "dirty, sanitary, clean, tidy, leaky, warm, dusky, bright, moldy, fusty, "
        "foul smelly, neat"
    ),
    "amenity": (
        "Amenity, payment method, bill issued, location, transportation, safety, "
        "security, lock, fire extinguisher, public service, spa, parking, traffic, "
        "place, site, surrounding, vicinity, travel, subway, bus stop, business "
        "district, city center, convenient, quick, well-suited"
    ),
    "service": (
        "Service, restaurant service, breakfast, food, beverage, staff, helpful, "
        "friendly, care, room services, booking services, room appointment, "
        "attitude, customer service, manager, front desk, proprietor, quality, "
        "cleaning, reception, sweeping, make up, waiter, service quality, work "
        "efficiency, passionate, caring, patiently, indifferent, considerate, "
        "thoughtful, kind, observant, meticulous, cordial"
    ),
    "experience": (
        "Atmosphere, noisy, quiet, relaxing, fresh, elegant, lovely, view, "
        "panorama, scenery, cultural, highlight, seaview, view, vision, "
        "satisfaction, hype, fame, promise, deliver, unsatisfaction, over-rated, "
        "overpriced, worth, value, price, fee, room rate, room price, "
        "cost-performance, entirety, in short, hotel, apartment, expensive, cheap, "
        "cost-effective, price increase, discount, concessional"
    ),
    "loyalty": (
        "Loyalty, revisit, back, recommend, suggest, again, stay away, never "
        "again, once is enough, stay elsewhere"
    ),
}

VALID_ASPECTS = list(KEY_ASPECTS) + ["other"]

TAXONOMY_BLOCK = "\n".join(f"- {k}: {v}" for k, v in KEY_ASPECTS.items())

SYSTEM_PROMPT = f"""You label hotel-review topics for a study of Vietnamese coastal hotels.
Reviews are in Vietnamese or English; all labels you produce are English snake_case.

Assign 1 to 3 aspects from this fixed taxonomy, ordered by dominance (most dominant
first). The keyword lists are semantic anchors describing what each aspect covers:
{TAXONOMY_BLOCK}

Multi-aspect rules:
- Each aspect gets its OWN sentiment (positive | negative | neutral). A topic can be
  service:positive AND facility:negative at the same time.
- Each aspect gets a weight in (0, 1]; weights across the topic MUST sum to 1.0.
  A single-aspect topic has one entry with weight 1.0.
- Only add a 2nd or 3rd aspect when it is genuinely present in the cluster. Most
  clean topic clusters are single-aspect - do not pad.

Assignment guidance:
- Physical things and their condition -> facility. People performing (or failing) work -> service.
- Landscape/scenery as a physical property attribute -> facility; the felt impression,
  seaview enjoyment, or value-for-money judgment -> experience.
- Anything about price, worth, cost-performance -> experience.
- Stated intent to return, recommend, or avoid -> include loyalty; if the cluster also
  gives the reason (a facility/service issue), include that aspect too.
- If nothing fits, use a single "other" aspect.

Within each aspect, generate sub_aspects: 1 to 3 free-form, finer-grained sub-labels.
Each sub_aspect MUST be 1-2 words, English, snake_case (e.g. "bed_comfort", "seaview",
"staff_friendliness", "value_for_money", "will_return"). evidence_keywords are words
from the topic vocabulary or excerpts supporting THAT aspect specifically."""

ASPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "key_aspect": {"type": "string", "enum": VALID_ASPECTS},
        "weight": {"type": "number"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "sub_aspects": {"type": "array", "items": {"type": "string"}},
        "evidence_keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["key_aspect", "weight", "sentiment", "sub_aspects", "evidence_keywords"],
    "additionalProperties": False,
}

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_id": {"type": "integer"},
        "aspects": {"type": "array", "items": ASPECT_SCHEMA},
        "confidence": {"type": "number"},
        "short_reason": {"type": "string"},
    },
    "required": ["topic_id", "aspects", "confidence", "short_reason"],
    "additionalProperties": False,
}


# ===========================================================================
# DuckDB schema + data access
# ===========================================================================

def ensure_label_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Add silver-label columns to TOPIC_LABELS and create TOPIC_ASPECTS."""
    con.execute("ALTER TABLE TOPIC_LABELS ADD COLUMN IF NOT EXISTS n_aspects INTEGER")
    con.execute("ALTER TABLE TOPIC_LABELS ADD COLUMN IF NOT EXISTS confidence FLOAT")
    con.execute("ALTER TABLE TOPIC_LABELS ADD COLUMN IF NOT EXISTS short_reason VARCHAR")
    con.execute("ALTER TABLE TOPIC_LABELS ADD COLUMN IF NOT EXISTS labeled_at TIMESTAMP")
    con.execute("""
        CREATE TABLE IF NOT EXISTS TOPIC_ASPECTS (
            run_id            VARCHAR  NOT NULL,
            topic_id          INTEGER  NOT NULL,
            key_aspect        VARCHAR  NOT NULL,
            is_primary        BOOLEAN  NOT NULL,
            weight            FLOAT    NOT NULL,
            sentiment         VARCHAR,
            sub_aspects       VARCHAR,
            evidence_keywords VARCHAR,
            PRIMARY KEY (run_id, topic_id, key_aspect)
        )
    """)


def fetch_topics(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    only_unlabeled: bool = True,
) -> list[tuple[int, str]]:
    """Return [(topic_id, top_words), ...] for one run, outliers excluded."""
    where = "AND n_aspects IS NULL" if only_unlabeled else ""
    rows = con.execute(f"""
        SELECT topic_id, top_words
        FROM TOPIC_LABELS
        WHERE run_id = ? AND topic_id != -1 {where}
        ORDER BY topic_id
    """, [run_id]).fetchall()
    return [(int(t), w or "") for t, w in rows]


def sample_docs(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    topic_id: int,
    k: int = N_SAMPLE_DOCS,
) -> list[str]:
    """Deterministically sample up to k representative raw review texts."""
    rows = con.execute("""
        SELECT r.review_text
        FROM REVIEW_TOPICS rt
        JOIN REVIEW_DATA r ON rt.review_id = r.review_id
        WHERE rt.run_id = ? AND rt.topic_id = ?
          AND r.review_text IS NOT NULL AND length(r.review_text) >= 60
        ORDER BY hash(rt.review_id)
        LIMIT ?
    """, [run_id, topic_id, k]).fetchall()
    return [r[0] for r in rows]


# ===========================================================================
# Batch request building + result parsing
# ===========================================================================

def make_custom_id(run_id: str, topic_id: int) -> str:
    return f"{run_id}__t{topic_id}"


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    run_id, tid = custom_id.rsplit("__t", 1)
    return run_id, int(tid)


def build_user_prompt(topic_id: int, top_words: str, docs: list[str]) -> str:
    words = ", ".join(w.strip() for w in top_words.split(",")[:N_TOP_WORDS])
    docs_block = "\n".join(f"- {d[:DOC_MAX_CHARS]}" for d in docs[:N_SAMPLE_DOCS])
    return f"""Topic ID: {topic_id}

Topic top words:
{words}

Representative review excerpts:
{docs_block}

Label this topic."""


def build_request(run_id: str, topic_id: int, top_words: str, docs: list[str]) -> dict:
    """One Batches-API request dict for one topic."""
    return {
        "custom_id": make_custom_id(run_id, topic_id),
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": build_user_prompt(topic_id, top_words, docs)}
            ],
            "output_config": {
                "format": {"type": "json_schema", "schema": LABEL_SCHEMA}
            },
        },
    }


_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def _snake(s: str) -> str:
    return _SNAKE_RE.sub("_", s.strip().lower()).strip("_")


def parse_label(raw_text: str) -> dict:
    """Parse + post-process one label: cap 3 aspects, renormalize weights,
    snake_case sub_aspects, clamp confidence, dedupe repeated aspects."""
    label = json.loads(raw_text)

    seen: set[str] = set()
    aspects = []
    for a in label["aspects"]:
        if a["key_aspect"] in seen:
            continue
        seen.add(a["key_aspect"])
        aspects.append(a)
    aspects = aspects[:3]
    if not aspects:
        raise ValueError("label has no aspects")

    total_w = sum(max(float(a.get("weight", 0)), 0.0) for a in aspects)
    for a in aspects:
        a["weight"] = (max(float(a.get("weight", 0)), 0.0) / total_w) if total_w > 0 else 1.0 / len(aspects)
        a["sub_aspects"] = [s for s in (_snake(x) for x in a.get("sub_aspects", [])) if s][:3]

    label["aspects"] = aspects
    label["confidence"] = min(max(float(label.get("confidence", 0.0)), 0.0), 1.0)
    return label


def write_label(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    topic_id: int,
    label: dict,
) -> None:
    """Idempotently write one parsed label to TOPIC_LABELS + TOPIC_ASPECTS."""
    aspects = label["aspects"]
    con.execute("""
        UPDATE TOPIC_LABELS
        SET n_aspects = ?, confidence = ?, short_reason = ?, labeled_at = now()
        WHERE run_id = ? AND topic_id = ?
    """, [len(aspects), label["confidence"], label["short_reason"], run_id, topic_id])

    con.execute(
        "DELETE FROM TOPIC_ASPECTS WHERE run_id = ? AND topic_id = ?",
        [run_id, topic_id],
    )
    for rank, a in enumerate(aspects):
        con.execute("""
            INSERT INTO TOPIC_ASPECTS
              (run_id, topic_id, key_aspect, is_primary, weight,
               sentiment, sub_aspects, evidence_keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id, topic_id, a["key_aspect"], rank == 0, a["weight"],
            a["sentiment"],
            json.dumps(a["sub_aspects"], ensure_ascii=False),
            json.dumps(a.get("evidence_keywords", []), ensure_ascii=False),
        ])


def is_language_run(run_id: str) -> bool:
    """Only per-language runs are valid (coast_band_A_en, year_2020_vi, ...).

    Mixed-language run_ids (coast_band_A, year_2020) are stale and excluded
    from the analysis entirely.
    """
    return run_id.endswith(("_en", "_vi"))


def current_run_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Per-language run ids that have both a checkpoint on disk and rows in
    TOPIC_LABELS. Stale mixed-language runs are filtered out."""
    on_disk = {p.stem for p in (ROOT / "checkpoints").glob("*.pkl")}
    in_db = {r[0] for r in con.execute("SELECT DISTINCT run_id FROM TOPIC_LABELS").fetchall()}
    return sorted(r for r in on_disk & in_db if is_language_run(r))
