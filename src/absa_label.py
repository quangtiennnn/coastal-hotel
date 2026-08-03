"""
absa_label.py
=============
Review-level ABSA silver-labeling (SENTENCES_LEVEL_PROPOSAL.md).

Design (bottom-up, two-tier):
- The LLM emits per review: sub_aspect (FREE string, seeded by the 34
  topic-level sub_aspects) + sentiment (hard enum) + evidence (verbatim
  substring). It does NOT emit key_aspect.
- key_aspect (facility/amenity/service/experience/loyalty/other) is DERIVED
  post-hoc via the sub_aspect -> key_aspect map inverted from
  llm_label.SUB_ASPECT_CHOICES. Unmapped sub_aspects -> 'other'
  (raw string retained in sub_aspect_raw).

Model:     claude-sonnet-5 via the Anthropic Message Batches API (50% off;
           intro pricing $1/$5 per MTok batched through 2026-08-31)
Batching:  N_REVIEWS_PER_REQUEST reviews per batch request (input-token lever)
Output:    enforced JSON via output_config.format (json_schema)
Storage:   REVIEW_ASPECTS (one row per extracted aspect span) +
           REVIEW_ASPECT_ROLLUP (per-review primary sentiment per macro
           aspect; REVIEW_DATA is a VIEW so roll-up columns cannot live there)

Used by:   src/absa_sample.py, src/absa_label_submit.py,
           src/absa_label_retrieve.py, src/absa_bridge.py,
           notebooks/23_absa_review_test.ipynb
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

# Reuse the topic-level taxonomy verbatim (proposal: import, don't redefine)
from llm_label import (  # noqa: F401  (get_client re-exported for the scripts)
    BATCH_DIR,
    DB_PATH,
    SUB_ASPECT_CHOICES,
    get_client,
)

ROOT = Path(__file__).parent.parent

MODEL = "claude-sonnet-5"
# Observed output: ~550-650 tokens PER review (JSON + evidence spans), so a
# 10-review chunk needs ~6.5k. 12k gives headroom for aspect-dense reviews;
# billing is per actual token, so a high ceiling costs nothing extra.
MAX_TOKENS = 12000
N_REVIEWS_PER_REQUEST = 10   # max reviews per batch request
MAX_CHUNK_CHARS = 5000       # max total review chars per request (output guard)
REVIEW_MAX_CHARS = 1500      # truncate pathological reviews in the prompt

LABEL_TIER = "silver"  # sub_aspects have no gold; key_aspect is gold-validated

# ---------------------------------------------------------------------------
# Derivation map: sub_aspect -> key_aspect (inverted SUB_ASPECT_CHOICES)
# ---------------------------------------------------------------------------

SUB_TO_KEY: dict[str, str] = {
    sub: key for key, subs in SUB_ASPECT_CHOICES.items() for sub in subs
}
KEY_ASPECTS = list(SUB_ASPECT_CHOICES) + ["other"]  # 5 + other
SENTIMENTS = ["positive", "negative", "neutral"]

_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def snake(s: str) -> str:
    return _SNAKE_RE.sub("_", s.strip().lower()).strip("_")


def derive_key_aspect(sub_aspect_norm: str) -> str:
    """Map a normalized sub_aspect onto the 5 macro aspects; unmapped -> other."""
    return SUB_TO_KEY.get(sub_aspect_norm, "other")


# ---------------------------------------------------------------------------
# Prompt (see SENTENCES_LEVEL_PROPOSAL.md - Sample labeling prompt)
# ---------------------------------------------------------------------------

_PREFERRED_BLOCK = "\n".join(
    f"  {aspect}: {', '.join(subs)}" for aspect, subs in SUB_ASPECT_CHOICES.items()
)

SYSTEM_PROMPT = f"""You extract aspect-based sentiment from hotel reviews for a study of Vietnamese
coastal hotels. Reviews are in Vietnamese or English; you always output English
snake_case sub_aspects and English sentiment labels.

For each review given, find every distinct thing the guest evaluates. For each,
emit one object with:
  - sub_aspect: a short snake_case tag for the specific thing evaluated.
      PREFER a tag from the list below when one fits. Only invent a new
      snake_case tag when none of them do. Do NOT output a key_aspect.
  - sentiment: exactly one of positive | negative | neutral.
  - evidence: the shortest VERBATIM substring of the review that supports it
      (copy the exact characters, including the original language/diacritics).

Preferred sub_aspects (reuse whenever they fit):
{_PREFERRED_BLOCK}

Rules:
  - Emit ONLY aspects actually discussed. Do not emit "not_mentioned" rows.
  - The same sub_aspect may appear multiple times with different sentiment and
    evidence (e.g. room clean but bathroom dirty -> two rows).
  - evidence MUST be an exact substring of the review text - never paraphrase.
  - If a review evaluates nothing, return an empty "aspects" array for it.
  - Return one result object per input review, in the same order, with the
    review_id copied exactly."""

# LLM output schema: NO key_aspect (derived downstream); sub_aspect is a free
# string (the 34 live in the prompt as a preference, not an enum); sentiment
# is a hard enum; evidence validated as substring at retrieve time.
ASPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_aspect": {"type": "string"},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "evidence": {"type": "string"},
    },
    "required": ["sub_aspect", "sentiment", "evidence"],
    "additionalProperties": False,
}

REVIEW_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "review_id": {"type": "string"},
        "aspects": {"type": "array", "items": ASPECT_SCHEMA},
    },
    "required": ["review_id", "aspects"],
    "additionalProperties": False,
}

BATCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {"type": "array", "items": REVIEW_RESULT_SCHEMA},
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


def pack_reviews(
    reviews: list[tuple[str, str, str]],
    max_reviews: int = N_REVIEWS_PER_REQUEST,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[list[tuple[str, str, str]]]:
    """Pack (review_id, language, text) tuples into request-sized chunks.

    Two caps per chunk: review count AND total review chars (measured after
    the REVIEW_MAX_CHARS truncation the prompt applies). Long reviews produce
    more aspect rows -> more output tokens, so a chunk of long reviews carries
    fewer of them; this keeps every request's output comfortably under
    MAX_TOKENS regardless of how lengths are distributed. A single oversized
    review always gets its own chunk (its text is capped at REVIEW_MAX_CHARS,
    which is < max_chars).
    """
    chunks: list[list[tuple[str, str, str]]] = []
    current: list[tuple[str, str, str]] = []
    current_chars = 0
    for rid, lang, text in reviews:
        n = min(len(text), REVIEW_MAX_CHARS)
        if current and (len(current) >= max_reviews or current_chars + n > max_chars):
            chunks.append(current)
            current, current_chars = [], 0
        current.append((rid, lang, text))
        current_chars += n
    if current:
        chunks.append(current)
    return chunks


def alias(i: int) -> str:
    """Short positional review_id used in the prompt (r1..rN).

    Real review_ids are 100+ char base64 strings the model occasionally
    mis-transcribes (observed ~0.5% loss in wave 1). Aliases are trivially
    copyable; the manifest order maps them back to real ids at retrieve time.
    """
    return f"r{i + 1}"


def build_user_prompt(reviews: list[tuple[str, str, str]]) -> str:
    """reviews: [(review_id, language, review_text), ...] - ids are replaced
    by positional aliases in the prompt (see alias())."""
    blocks = []
    for i, (_rid, lang, text) in enumerate(reviews):
        text = " ".join(text.split())[:REVIEW_MAX_CHARS]
        blocks.append(f"review_id: {alias(i)}\nlanguage: {lang}\nreview: {text}")
    joined = "\n\n---\n\n".join(blocks)
    return f"Label the following {len(reviews)} reviews.\n\n{joined}"


def alias_map(review_ids: list[str]) -> dict[str, str]:
    """alias -> real review_id for one chunk (manifest order)."""
    return {alias(i): rid for i, rid in enumerate(review_ids)}


def build_request(chunk_id: str, reviews: list[tuple[str, str, str]]) -> dict:
    """One Batches-API request dict covering N reviews."""
    return {
        "custom_id": chunk_id,
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": build_user_prompt(reviews)}
            ],
            "output_config": {
                "format": {"type": "json_schema", "schema": BATCH_OUTPUT_SCHEMA}
            },
        },
    }


# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------

def ensure_absa_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ABSA_SAMPLE (
            review_id  VARCHAR NOT NULL,
            run_id     VARCHAR NOT NULL,   -- coast_band_{A,B,C}_{en,vi}
            topic_id   INTEGER NOT NULL,
            source     VARCHAR,            -- agoda / googlemaps
            language   VARCHAR,            -- en / vi
            split      VARCHAR,            -- train / val / test
            sampled_at TIMESTAMP,
            PRIMARY KEY (review_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS REVIEW_ASPECTS (
            review_id       VARCHAR NOT NULL,
            aspect_rank     INTEGER NOT NULL,  -- order emitted by the model
            sub_aspect_raw  VARCHAR NOT NULL,  -- exactly what the LLM emitted
            sub_aspect      VARCHAR NOT NULL,  -- snake_cased / normalized
            key_aspect      VARCHAR NOT NULL,  -- DERIVED via SUB_TO_KEY; unmapped -> other
            sentiment       VARCHAR NOT NULL,  -- positive/negative/neutral
            evidence        VARCHAR,           -- verbatim substring of review_text
            evidence_valid  BOOLEAN,           -- substring check result
            label_model     VARCHAR,
            labeled_at      TIMESTAMP,
            PRIMARY KEY (review_id, aspect_rank)
        )
    """)
    # REVIEW_DATA is a VIEW -> roll-up lives in its own table, not asp5_* columns
    con.execute("""
        CREATE TABLE IF NOT EXISTS REVIEW_ASPECT_ROLLUP (
            review_id       VARCHAR NOT NULL PRIMARY KEY,
            asp5_facility   VARCHAR,
            asp5_amenity    VARCHAR,
            asp5_service    VARCHAR,
            asp5_experience VARCHAR,
            asp5_loyalty    VARCHAR,
            n_aspects       INTEGER,
            label_model     VARCHAR,
            labeled_at      TIMESTAMP
        )
    """)


# ---------------------------------------------------------------------------
# Parsing / validation / writes
# ---------------------------------------------------------------------------

def parse_batch_result(raw_text: str) -> list[dict]:
    """Parse one batch response -> [{review_id, aspects: [...]}, ...]."""
    data = json.loads(raw_text)
    return data["reviews"]


def find_evidence(text: str, evidence: str, start: int = 0) -> tuple[int, int] | None:
    """Locate `evidence` in `text`, tolerant of whitespace differences.

    The prompt collapses runs of whitespace before the model sees the review,
    so a verbatim copy of what the model saw may differ from the raw text by
    whitespace only (e.g. double spaces, newlines). Exact match first, then a
    regex with \\s+ between tokens. Returns (start, end) or None.
    """
    if not evidence:
        return None
    idx = text.find(evidence, start)
    if idx != -1:
        return idx, idx + len(evidence)
    tokens = evidence.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.compile(pattern).search(text, start)
    return (m.start(), m.end()) if m else None


def normalize_aspects(aspects: list[dict], review_text: str) -> list[dict]:
    """snake_case, derive key_aspect, validate evidence (whitespace-tolerant)."""
    out = []
    for a in aspects:
        raw = str(a.get("sub_aspect", "")).strip()
        norm = snake(raw)
        if not norm:
            continue
        evidence = a.get("evidence") or ""
        out.append({
            "sub_aspect_raw": raw,
            "sub_aspect": norm,
            "key_aspect": derive_key_aspect(norm),
            "sentiment": a["sentiment"],
            "evidence": evidence,
            "evidence_valid": find_evidence(review_text, evidence) is not None,
        })
    return out


def rollup(aspects: list[dict]) -> dict[str, str | None]:
    """One primary sentiment per macro aspect: majority, ties -> neutral."""
    cols: dict[str, str | None] = {k: None for k in SUB_ASPECT_CHOICES}
    for key in cols:
        sents = [a["sentiment"] for a in aspects if a["key_aspect"] == key]
        if not sents:
            continue
        counts = {s: sents.count(s) for s in set(sents)}
        best = max(counts.values())
        winners = [s for s, c in counts.items() if c == best]
        cols[key] = winners[0] if len(winners) == 1 else "neutral"
    return cols


def write_review_aspects(
    con: duckdb.DuckDBPyConnection,
    review_id: str,
    aspects: list[dict],
    label_model: str = MODEL,
) -> None:
    """Idempotently write one review's aspects + roll-up."""
    con.execute("DELETE FROM REVIEW_ASPECTS WHERE review_id = ?", [review_id])
    for rank, a in enumerate(aspects):
        con.execute("""
            INSERT INTO REVIEW_ASPECTS
              (review_id, aspect_rank, sub_aspect_raw, sub_aspect, key_aspect,
               sentiment, evidence, evidence_valid, label_model, labeled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, now())
        """, [
            review_id, rank, a["sub_aspect_raw"], a["sub_aspect"],
            a["key_aspect"], a["sentiment"], a["evidence"],
            a["evidence_valid"], label_model,
        ])

    cols = rollup(aspects)
    con.execute("DELETE FROM REVIEW_ASPECT_ROLLUP WHERE review_id = ?", [review_id])
    con.execute("""
        INSERT INTO REVIEW_ASPECT_ROLLUP
          (review_id, asp5_facility, asp5_amenity, asp5_service,
           asp5_experience, asp5_loyalty, n_aspects, label_model, labeled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())
    """, [
        review_id, cols["facility"], cols["amenity"], cols["service"],
        cols["experience"], cols["loyalty"], len(aspects), label_model,
    ])


def fetch_review_texts(
    con: duckdb.DuckDBPyConnection, review_ids: list[str]
) -> dict[str, tuple[str, str]]:
    """review_id -> (language, review_text)."""
    if not review_ids:
        return {}
    rows = con.execute("""
        SELECT review_id, language, review_text
        FROM REVIEW_DATA
        WHERE review_id IN (SELECT unnest(?::VARCHAR[]))
    """, [review_ids]).fetchall()
    return {r[0]: (r[1] or "", r[2] or "") for r in rows}


def unlabeled_sample_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Sampled review_ids not yet processed by a retrieve.

    Keyed on REVIEW_ASPECT_ROLLUP (one row per processed review, including
    legitimately zero-aspect reviews) - NOT on REVIEW_ASPECTS, which would
    re-send aspect-less reviews forever.
    """
    rows = con.execute("""
        SELECT s.review_id
        FROM ABSA_SAMPLE s
        LEFT JOIN REVIEW_ASPECT_ROLLUP r USING (review_id)
        WHERE r.review_id IS NULL
        ORDER BY s.review_id
    """).fetchall()
    return [r[0] for r in rows]
