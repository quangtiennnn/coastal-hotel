# Proposal V2: Coastal-Distance & Temporal Topic Analysis with Claude-Labeled Fixed Aspects

Supersedes `PROPOSAL_COASTAL_TOPIC_ANALYSIS.md`. Three changes from V1:

1. **Fixed 5-aspect taxonomy** — aspects come from a closed set of 5 hotel-review dimensions (Facility / Amenity / Service / Experience / Loyalty), each anchored by an explicit keyword list. `sub_aspect` remains free-form (1–2 words).
2. **Multi-aspect labels with per-aspect sentiment** — a topic that spans aspects (e.g. "breakfast room was dirty but staff were lovely") is labeled with **1–3 aspects, each carrying its own sentiment, sub_aspects, and weight**. One aspect is marked primary so single-line-per-aspect charts still work.
3. **Claude API replaces Gemini** — silver-labeling runs on `claude-sonnet-4-6` via the **Anthropic Message Batches API** (50% discount, structured JSON output). Cost estimate included at the end.

---

## Goal

Run two complementary BERTopic analyses on the unified review corpus (`REVIEW_DATA`, 251 k rows, en + vi), then visualize how **aspect** distributions shift:

1. **Across coastline-distance bands** — do beachfront hotels attract different feedback than inland ones?
2. **Across time (2018–2024)** — are aspect proportions drifting year over year?

Both analyses reuse the pre-computed embeddings in `REVIEW_EMBEDDINGS` (no re-encoding). **Every slice is fitted per language** — Vietnamese and English reviews get separate models (`coast_band_A_en` / `coast_band_A_vi`, `year_2020_en` / `year_2020_vi`, …) because tokenization, stopwords, and topic vocabularies differ completely between the two. Mixed-language run_ids (`coast_band_A`, `year_2020`) are **not part of the analysis** and have been removed from the database. That gives **20 runs total**: 3 bands × 2 languages + 7 years × 2 languages.

Each run is fitted independently so its topic space is clean; each fitted model is saved as a checkpoint. After fitting, every non-outlier topic goes through a **Claude-assisted silver-label step** that maps raw BERTopic keyword clusters onto the fixed taxonomy. Labels are English regardless of input language, so en and vi runs become directly comparable at the aspect level — the charts aggregate both languages per band/year. The silver labels are the unit of analysis for the line graphs.

---

## Distance Bands

The `distance2coastline` column in `REVIEW_DATA` stores distance in **km**.

| Band | Label | SQL filter |
| --- | --- | --- |
| A | Beachfront (`< 0.1 km`) | `distance2coastline < 0.1` |
| B | Near-coast (`0.1 – 0.5 km`) | `distance2coastline >= 0.1 AND distance2coastline < 0.5` |
| C | Inland (`≥ 0.5 km`) | `distance2coastline >= 0.5` |

---

## Fixed `key_aspect` Taxonomy

The LLM assigns **1 to 3 aspects** from the 5 below (or `other` when nothing fits). The keyword lists are included verbatim in the labeling prompt as anchors — they define the semantic boundary of each aspect, they are *not* a string-matching filter.

| key_aspect | Anchor keywords |
| --- | --- |
| `facility` | Facility, room, furnishings, bathroom, charging, reception area, restaurant facilities, public equipment, gym, pool, elevator, bed, mattress, water, electricity, configuration, home appliances, article, supplies, utensils, decoration, air conditioner, infrastructure, environment, sanitation, sound insulation, ventilation, natural lighting, landscape, scenery, new, old, complete, antiquated, shabby, wet, dark, dirty, sanitary, clean, tidy, leaky, warm, dusky, bright, moldy, fusty, foul smelly, neat |
| `amenity` | Amenity, payment method, bill issued, location, transportation, safety, security, lock, fire extinguisher, public service, spa, parking, traffic, place, site, surrounding, vicinity, travel, subway, bus stop, business district, city center, convenient, quick, well-suited |
| `service` | Service, restaurant service, breakfast, food, beverage, staff, helpful, friendly, care, room services, booking services, room appointment, attitude, customer service, manager, front desk, proprietor, quality, cleaning, reception, sweeping, make up, waiter, service quality, work efficiency, passionate, caring, patiently, indifferent, considerate, thoughtful, kind, observant, meticulous, cordial |
| `experience` | Atmosphere, noisy, quiet, relaxing, fresh, elegant, lovely, view, panorama, scenery, cultural, highlight, seaview, view, vision, satisfaction, hype, fame, promise, deliver, unsatisfaction, over-rated, overpriced, worth, value, price, fee, room rate, room price, cost-performance, entirety, in short, hotel, apartment, expensive, cheap, cost-effective, price increase, discount, concessional |
| `loyalty` | Loyalty, revisit, back, recommend, suggest, again, stay away, never again, once is enough, stay elsewhere |

### Multi-aspect rules baked into the prompt

- Assign **1–3 aspects**, ordered by dominance; the first is the **primary** aspect.
- Each aspect carries its **own sentiment** — a topic can be `service: positive` + `facility: negative` at the same time. No more lossy `mixed` topic-level sentiment.
- Each aspect carries a **weight** (0–1, all weights sum to 1.0) reflecting how much of the cluster's content belongs to it. Single-aspect topics get weight 1.0.
- Only add a second/third aspect when it's genuinely present in the cluster — don't pad. Most clean BERTopic clusters are single-aspect.

Assignment guidance (boundary cases):

- **Physical thing vs. people doing things**: a dirty bathroom is `facility`; a maid who never cleaned is `service`. If a cluster has both, emit both aspects.
- **`scenery`/`view` appear in two lists**: physical landscape *of the property* → `facility`; the felt impression / seaview / value-for-money judgment → `experience`.
- **Price & worth** always → `experience` (it covers experience-value).
- **Intent to return, recommend, or avoid** → include `loyalty` as an aspect; if the cluster also explains *why* (a facility or service reason), include that reason aspect too.
- Topics that genuinely fit none → single aspect `"other"` (excluded from chart denominators, kept in the DB).

### `sub_aspect` — free generation, 1–2 words, nested per aspect

Within **each** assigned aspect the LLM generates 1–3 finer-grained sub-aspects, **each 1–2 words, English snake_case**, regardless of review language. Examples:

- `facility` → `["bed_comfort", "pool", "old_building"]`
- `service` → `["staff_friendliness", "checkin_speed"]`
- `experience` → `["seaview", "value_for_money", "noise"]`
- `loyalty` → `["will_return", "recommend"]`

---

## DuckDB Tables

Three tables in `hotel_reviews.db`. Because a topic can now carry multiple aspects, the aspect payload is **normalized into its own table** (`TOPIC_ASPECTS`) instead of living as columns on `TOPIC_LABELS` — this keeps the chart queries plain SQL joins, no JSON unpacking.

### `TOPIC_LABELS` — one row per topic (topic-level metadata)

```sql
CREATE TABLE TOPIC_LABELS (
    run_id       VARCHAR  NOT NULL,   -- "coast_band_A_en".."coast_band_C_vi", "year_2018_en".."year_2024_vi"
    topic_id     INTEGER  NOT NULL,   -- BERTopic topic number (-1 = outlier)
    top_words    VARCHAR,             -- comma-joined representation keywords
    n_aspects    INTEGER,             -- 1..3
    confidence   FLOAT,               -- LLM confidence 0-1 (whole label)
    short_reason VARCHAR,             -- one-sentence justification
    n_docs       INTEGER,
    PRIMARY KEY (run_id, topic_id)
);
```

### `TOPIC_ASPECTS` — one row per (topic, aspect)

```sql
CREATE TABLE TOPIC_ASPECTS (
    run_id            VARCHAR  NOT NULL,
    topic_id          INTEGER  NOT NULL,
    key_aspect        VARCHAR  NOT NULL,   -- facility | amenity | service | experience | loyalty | other
    is_primary        BOOLEAN  NOT NULL,   -- exactly one TRUE per (run_id, topic_id)
    weight            FLOAT    NOT NULL,   -- 0-1; weights sum to 1.0 per topic
    sentiment         VARCHAR,             -- positive | negative | neutral  (per aspect!)
    sub_aspects       VARCHAR,             -- JSON array of free-form 1-2 word sub-labels
    evidence_keywords VARCHAR,             -- JSON array of supporting words for THIS aspect
    PRIMARY KEY (run_id, topic_id, key_aspect)
);
```

### `REVIEW_TOPICS`

```sql
CREATE TABLE REVIEW_TOPICS (
    run_id     VARCHAR  NOT NULL,
    review_id  VARCHAR  NOT NULL,
    topic_id   INTEGER  NOT NULL,
    prob       FLOAT,
    PRIMARY KEY (run_id, review_id)
);
```

---

## Claude-Assisted Silver-Label Step

### Design

- **Model**: `claude-sonnet-4-6` — multi-aspect classification with per-aspect sentiment is comfortably within Sonnet capability; Opus pricing isn't justified here.
- **Transport**: **Message Batches API** (`client.messages.batches.create`) — 50% off all token usage, up to 100 k requests per batch, most batches finish within 1 hour. One batch request per topic, one batch submission per run (or one combined batch for all 10 runs).
- **Output enforcement**: `output_config.format` with a strict JSON schema — guarantees parseable JSON, no markdown fences, valid `key_aspect` enum.
- **Two scripts, not one**: `submit` (fires the batch, persists `batch_id`, PC can go offline) and `retrieve` (polls/collects results later, writes `TOPIC_LABELS`). Batch results stay available for 29 days.
- **Validate before scaling**: test 2–3 topics through the standard (non-batch) `messages.create` endpoint first to confirm prompt + schema parsing, then fire the full batch.

### Per-topic input

Each batch request carries:

- The fixed taxonomy + disambiguation rules (shared prompt, ~800 tokens)
- The topic's **top representation words** (up to 10)
- Up to **5 representative review excerpts** (≤ 120 tokens each), sampled from the cluster

### Expected JSON output per topic

Single-aspect topic (the common case):

```json
{
  "topic_id": 3,
  "aspects": [
    {
      "key_aspect": "facility",
      "weight": 1.0,
      "sentiment": "positive",
      "sub_aspects": ["bed_comfort", "seaview_room"],
      "evidence_keywords": ["phòng", "giường", "êm", "view", "rộng"]
    }
  ],
  "confidence": 0.88,
  "short_reason": "Cluster centers on room/bed quality and the in-room sea view, framed positively."
}
```

Multi-aspect topic with split sentiment:

```json
{
  "topic_id": 17,
  "aspects": [
    {
      "key_aspect": "service",
      "weight": 0.6,
      "sentiment": "positive",
      "sub_aspects": ["staff_friendliness", "breakfast_service"],
      "evidence_keywords": ["nhân viên", "nhiệt tình", "thân thiện"]
    },
    {
      "key_aspect": "facility",
      "weight": 0.4,
      "sentiment": "negative",
      "sub_aspects": ["old_building", "bathroom"],
      "evidence_keywords": ["cũ", "xuống cấp", "nhà vệ sinh"]
    }
  ],
  "confidence": 0.81,
  "short_reason": "Reviews praise friendly staff while complaining the building and bathrooms are aged."
}
```

The first element of `aspects` is the primary aspect (`is_primary = TRUE` on insert).

### Labeling code

```python
# src/llm_label.py

import json
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

MODEL = "claude-sonnet-4-6"

KEY_ASPECTS = {
    "facility": "Facility, room, furnishings, bathroom, charging, reception area, restaurant facilities, public equipment, gym, pool, elevator, bed, mattress, water, electricity, configuration, home appliances, article, supplies, utensils, decoration, air conditioner, infrastructure, environment, sanitation, sound insulation, ventilation, natural lighting, landscape, scenery, new, old, complete, antiquated, shabby, wet, dark, dirty, sanitary, clean, tidy, leaky, warm, dusky, bright, moldy, fusty, foul smelly, neat",
    "amenity": "Amenity, payment method, bill issued, location, transportation, safety, security, lock, fire extinguisher, public service, spa, parking, traffic, place, site, surrounding, vicinity, travel, subway, bus stop, business district, city center, convenient, quick, well-suited",
    "service": "Service, restaurant service, breakfast, food, beverage, staff, helpful, friendly, care, room services, booking services, room appointment, attitude, customer service, manager, front desk, proprietor, quality, cleaning, reception, sweeping, make up, waiter, service quality, work efficiency, passionate, caring, patiently, indifferent, considerate, thoughtful, kind, observant, meticulous, cordial",
    "experience": "Atmosphere, noisy, quiet, relaxing, fresh, elegant, lovely, view, panorama, scenery, cultural, highlight, seaview, view, vision, satisfaction, hype, fame, promise, deliver, unsatisfaction, over-rated, overpriced, worth, value, price, fee, room rate, room price, cost-performance, entirety, in short, hotel, apartment, expensive, cheap, cost-effective, price increase, discount, concessional",
    "loyalty": "Loyalty, revisit, back, recommend, suggest, again, stay away, never again, once is enough, stay elsewhere",
}

ASPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "key_aspect": {
            "type": "string",
            "enum": ["facility", "amenity", "service", "experience", "loyalty", "other"],
        },
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
# Note: structured outputs can't enforce array length or numeric ranges —
# the 1-3 aspect cap and weight normalization are enforced by prompt + post-processing.

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
  clean topic clusters are single-aspect — do not pad.

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


def build_request(run_id: str, topic_id: int, top_words: list[str], sample_docs: list[str]) -> Request:
    docs_block = "\n".join(f"- {d[:480]}" for d in sample_docs[:5])  # ~120 tokens each
    user_prompt = f"""Topic ID: {topic_id}

Topic top words:
{", ".join(top_words[:10])}

Representative review excerpts:
{docs_block}

Label this topic."""

    return Request(
        custom_id=f"{run_id}__topic_{topic_id}",
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        ),
    )
```

### Submit script

```python
# src/llm_label_submit.py

import json
from pathlib import Path

def submit_run(run_id: str, topic_model, docs: list[str], topics: list[int]) -> str:
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1]

    requests = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        idx = [i for i, t in enumerate(topics) if t == tid][:5]
        requests.append(build_request(run_id, tid, row["Representation"], [docs[i] for i in idx]))

    batch = client.messages.batches.create(requests=requests)
    Path(f"batches/{run_id}.batch_id").write_text(batch.id)
    print(f"[{run_id}] submitted {len(requests)} topics -> {batch.id}")
    return batch.id
```

### Retrieve script

```python
# src/llm_label_retrieve.py

import duckdb, json
from pathlib import Path

def retrieve_run(run_id: str) -> None:
    batch_id = Path(f"batches/{run_id}.batch_id").read_text().strip()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"[{run_id}] still {batch.processing_status} "
              f"({batch.request_counts.processing} processing) — try again later")
        return

    con = duckdb.connect("hotel_reviews.db")
    n_ok, n_err = 0, 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            n_err += 1
            print(f"  ! {result.custom_id}: {result.result.type}")
            continue
        msg = result.result.message
        label = json.loads(next(b.text for b in msg.content if b.type == "text"))
        tid = label["topic_id"]

        # post-process: cap at 3 aspects, renormalize weights to sum to 1.0
        aspects = label["aspects"][:3]
        total_w = sum(a["weight"] for a in aspects) or 1.0
        for a in aspects:
            a["weight"] = a["weight"] / total_w

        con.execute("""
            INSERT OR REPLACE INTO TOPIC_LABELS
              (run_id, topic_id, top_words, n_aspects, confidence, short_reason, n_docs)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [run_id, tid, None, len(aspects),
              label["confidence"], label["short_reason"], None])

        con.execute("DELETE FROM TOPIC_ASPECTS WHERE run_id = ? AND topic_id = ?", [run_id, tid])
        for rank, a in enumerate(aspects):
            con.execute("""
                INSERT INTO TOPIC_ASPECTS
                  (run_id, topic_id, key_aspect, is_primary, weight,
                   sentiment, sub_aspects, evidence_keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, tid, a["key_aspect"], rank == 0, a["weight"],
                a["sentiment"],
                json.dumps(a["sub_aspects"], ensure_ascii=False),
                json.dumps(a["evidence_keywords"], ensure_ascii=False),
            ])
        n_ok += 1
    con.close()
    print(f"[{run_id}] wrote {n_ok} labels ({n_err} errors)")
```

(`top_words` / `n_docs` are back-filled from the checkpointed model in the same script — omitted here for brevity.)

Post-processing: cap at 3 aspects + renormalize weights (in `retrieve`, above); lowercase + underscore-normalize `sub_aspects`; flag `confidence < 0.5` topics for manual review; `key_aspect = 'other'` excluded from chart denominators.

---

## Pipeline — Analysis 1: Distance-Band Models

**6 independent checkpoints** (3 bands × 2 languages): `coast_band_{A,B,C}_{en,vi}`.

```python
# pseudocode

BANDS = {
    "coast_band_A": "r.distance2coastline < 0.1",
    "coast_band_B": "r.distance2coastline >= 0.1 AND r.distance2coastline < 0.5",
    "coast_band_C": "r.distance2coastline >= 0.5",
}

for band, where_clause in BANDS.items():
    for lang in ("en", "vi"):
        run_id = f"{band}_{lang}"
        ckpt = Path(f"checkpoints/{run_id}.pkl")
        if ckpt.exists():
            topic_model = BERTopic.load(str(ckpt))
        else:
            df, docs, embeddings = load_from_duckdb(language=lang, extra_where=where_clause)
            topic_model = build_bertopic(min_cluster_size=30, min_topic_size=30, language=lang)
            topics, _ = topic_model.fit_transform(docs, embeddings)
            topic_model.save(str(ckpt), serialization="pickle", save_ctfidf=True)
            _write_to_duckdb(run_id, df, topics, topic_model)
```

> **Status: fitting is already done** — all 6 checkpoints exist and `TOPIC_LABELS` / `REVIEW_TOPICS` are populated. Actual topic counts: A_en 6, A_vi 5, B_en 15, B_vi 14, C_en 34, C_vi 135 → **209 coast topics**. Labeling is launched separately via `src/llm_label_submit.py`.

## Pipeline — Analysis 2: Year-Slice Models

**14 independent checkpoints** (7 years × 2 languages): `year_{2018..2024}_{en,vi}`. Identical loop with `load_from_duckdb(language=lang, min_year=year, max_year=year)`. Years with < 5 k reviews: drop `min_cluster_size` to 20.

> **Status: fitting is already done** — all 14 checkpoints exist; **316 year topics** in `TOPIC_LABELS`.

---

## Visualisation — Line Graphs

Both charts group by the 5 fixed aspects, so each chart is **exactly 5 lines** — directly comparable across bands and years. Multi-aspect topics contribute **fractionally via `weight`** (a review in a 0.6 service / 0.4 facility topic counts 0.6 toward service, 0.4 toward facility), so shares still sum to 100% per band/year. For a simpler view, filter `ta.is_primary` and count whole reviews instead.

### Chart 1 — Aspect share over distance bands (weighted)

```sql
SELECT
    CASE
        WHEN r.distance2coastline < 0.1  THEN 'Beachfront'
        WHEN r.distance2coastline < 0.5  THEN 'Near-coast'
        ELSE 'Inland'
    END                            AS band,
    ta.key_aspect,
    SUM(ta.weight) * 100.0 / SUM(SUM(ta.weight)) OVER (PARTITION BY band) AS pct
FROM REVIEW_TOPICS  rt
JOIN TOPIC_ASPECTS  ta ON rt.run_id = ta.run_id AND rt.topic_id = ta.topic_id
JOIN REVIEW_DATA    r  ON rt.review_id = r.review_id
WHERE rt.run_id LIKE 'coast_band_%'
  AND rt.topic_id != -1
  AND ta.key_aspect != 'other'
GROUP BY band, ta.key_aspect
ORDER BY band, ta.key_aspect;
```

### Chart 2 — Aspect share over years (2018–2024, weighted)

```sql
SELECT
    CAST(regexp_extract(rt.run_id, 'year_(\d{4})', 1) AS INTEGER) AS year,  -- run_id is year_YYYY_en / year_YYYY_vi
    ta.key_aspect,
    SUM(ta.weight) * 100.0 / SUM(SUM(ta.weight)) OVER (PARTITION BY year) AS pct
FROM REVIEW_TOPICS  rt
JOIN TOPIC_ASPECTS  ta ON rt.run_id = ta.run_id AND rt.topic_id = ta.topic_id
WHERE rt.run_id LIKE 'year_%'
  AND rt.topic_id != -1
  AND ta.key_aspect != 'other'
GROUP BY year, ta.key_aspect
ORDER BY year, ta.key_aspect;
```

### Chart 3 (new, enabled by per-aspect sentiment) — Negative share within each aspect

Because sentiment now lives on the (topic, aspect) pair, you can plot *how negative each aspect is* over time or bands — e.g. did `facility` complaints spike post-COVID while `service` stayed positive?

```sql
SELECT
    CAST(regexp_extract(rt.run_id, 'year_(\d{4})', 1) AS INTEGER) AS year,  -- run_id is year_YYYY_en / year_YYYY_vi
    ta.key_aspect,
    SUM(CASE WHEN ta.sentiment = 'negative' THEN ta.weight ELSE 0 END) * 100.0
        / SUM(ta.weight)                              AS pct_negative
FROM REVIEW_TOPICS  rt
JOIN TOPIC_ASPECTS  ta ON rt.run_id = ta.run_id AND rt.topic_id = ta.topic_id
WHERE rt.run_id LIKE 'year_%'
  AND rt.topic_id != -1
  AND ta.key_aspect != 'other'
GROUP BY year, ta.key_aspect
ORDER BY year, ta.key_aspect;
```

Bonus drill-down: within one aspect, plot the top-N `sub_aspects` over years (e.g. is `value_for_money` growing inside `experience`?).

Rendered in `notebooks/11_topic_coast_time.ipynb` (matplotlib, one colour per aspect, shared legend).

---

## API Cost Estimate

**Pricing** (`claude-sonnet-4-6`): standard $3.00 input / $15.00 output per MTok; **Batch API = 50% off → $1.50 / $7.50 per MTok**.

**Volume** (actual counts from `TOPIC_LABELS`, per-language runs only):

| Quantity | Actual |
| --- | --- |
| Model runs | 20 (3 bands × 2 langs + 7 years × 2 langs) |
| Coast topics (`coast_band_{A,B,C}_{en,vi}`) | **209** |
| Year topics (`year_{2018..2024}_{en,vi}`) | **316** |
| **Total topics ≈ requests** | **525** |
| Input tokens / request | ~1,600 (taxonomy + multi-aspect rules ~900, top words ~50, 5 excerpts ~600 — Vietnamese tokenizes heavier, margin included) |
| Output tokens / request | ~350 (strict JSON; multi-aspect topics emit 2–3 aspect objects) |

**Cost** (≈ $0.005 per topic at batch rates):

| Phase | Topics | Input cost | Output cost | Total |
| --- | --- | --- | --- | --- |
| Coast bands only (phase 1) | 209 | $0.50 | $0.55 | **≈ $1.05** |
| Year slices (phase 2) | 316 | $0.76 | $0.83 | **≈ $1.59** |
| Smoke test (notebook 17, 1-topic batch) | 1 | — | — | < $0.01 |
| **Everything** | **525** | $1.26 | $1.38 | **≈ $2.70** |

Sensitivity: one full re-run after prompt iteration doubles the phase you re-run. Budget **$3–6** end-to-end.

Notes:
- Prompt caching would normally cut the shared ~800-token prefix, but it's **below Sonnet 4.6's 2,048-token minimum cacheable prefix** and the Batch API already halves everything — not worth engineering.
- The whole job fits in **one or a few batches** (limit is 100 k requests/batch); expect results within ~1 hour, 24 h max, retrievable for 29 days.

---

## Files Affected

| File | Action |
| --- | --- |
| Model fitting (20 per-language runs) | **Already done** — `checkpoints/*.pkl` + `TOPIC_LABELS` / `REVIEW_TOPICS` populated; stale mixed-language runs deleted |
| `src/llm_label.py` | **Done** — taxonomy, prompt, JSON schema, batch-request builder, DuckDB schema + writes; per-language run_ids enforced |
| `src/llm_label_submit.py` | **Done** — submits one batch (rejects mixed run_ids), persists `batch_id` to `batches/` |
| `src/llm_label_retrieve.py` | **Done** — polls batch, parses results, writes `TOPIC_LABELS` + `TOPIC_ASPECTS` |
| `notebooks/17_silver_label_test.ipynb` | **Done** — 1-topic smoke test through the Batch API |
| `notebooks/11_topic_coast_time.ipynb` | Update — load from `TOPIC_ASPECTS`, draw the 3 charts + sub_aspect drill-down |
| `pyproject.toml` | `anthropic` already present — no change |

## What Does NOT Change

- `REVIEW_EMBEDDINGS` — same vectors, reused by both analyses
- `src/preprocessor.py`, preprocessing/embedding scripts — untouched
- `build_bertopic()` config — same UMAP / HDBSCAN / KeyBERT setup, only `min_cluster_size` tuned per slice
- `HOTEL`, `AGODA_REVIEW`, `GOOGLEMAPS_REVIEW`, `REVIEW_DATA` — untouched

---

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Topic spans two aspects (e.g. "breakfast room was dirty") | **Handled natively** — up to 3 aspects per topic, each with its own sentiment + weight; primary aspect flagged for simple views |
| LLM over-splits (pads 2nd/3rd aspects onto clean clusters) | Prompt explicitly says "do not pad"; audit: distribution of `n_aspects` should be heavily skewed to 1 — if not, tighten the prompt and re-run the affected batch |
| Weights don't sum to 1.0 / more than 3 aspects | Structured outputs can't enforce numeric ranges or array length — `retrieve` truncates to 3 and renormalizes weights |
| LLM drifts outside the 5-way enum | `output_config.format` json_schema with `enum` — invalid aspects are impossible by construction |
| `sub_aspects` spelling inconsistency across runs | 1–2 word snake_case constraint in prompt + post-process normalization; near-duplicate merge (e.g. `sea_view`/`seaview`) in notebook 11 |
| Small band/year slices → noisy topics | `min_cluster_size` 20 for slices < 10 k rows; annotate in charts |
| Same concept lands on different topic IDs per run | Fixed taxonomy normalizes across runs — charts use `key_aspect`, never raw `topic_id` |
| Batch errors / partial failures | `retrieve` logs per-request errors; failed `custom_id`s re-submitted in a small follow-up batch; `INSERT OR REPLACE` makes re-runs idempotent |
| PC offline during batch processing | Submit/retrieve split — results persist 29 days server-side |
| Low-confidence labels (`< 0.5`) | Excluded from charts, kept in `TOPIC_LABELS` for manual review |
| `distance2coastline` nulls | `WHERE distance2coastline IS NOT NULL` in band queries |
| Checkpoint `.pkl` bloat (~50–200 MB × 10) | `checkpoints/` gitignored; regenerable from `REVIEW_EMBEDDINGS` |
