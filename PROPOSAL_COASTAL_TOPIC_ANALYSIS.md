# Proposal: Coastal-Distance & Temporal Topic Analysis

## Goal

Run two complementary BERTopic analyses on the unified review corpus (`REVIEW_DATA`, 251 k rows, en + vi), then visualize how topic distributions shift:

1. **Across coastline-distance bands** — do beachfront hotels attract different feedback than inland ones?
2. **Across time (2018–2024)** — are topic proportions drifting year over year?

Both analyses share the same pre-computed embeddings already stored in `REVIEW_EMBEDDINGS`, so no re-encoding is needed. Each slice is fitted independently so its topic space is clean, and each fitted model is saved as a checkpoint.

---

## Distance Bands

The `distance2coastline` column in `REVIEW_DATA` stores distance in **km**.  
Three bands capture the gradient from beachfront to clearly inland:

| Band | Label | SQL filter |
| --- | --- | --- |
| A | Beachfront (`< 0.1 km`) | `distance2coastline < 0.1` |
| B | Near-coast (`0.1 – 0.5 km`) | `distance2coastline >= 0.1 AND distance2coastline < 0.5` |
| C | Inland (`≥ 0.5 km`) | `distance2coastline >= 0.5` |

---

## Two New DuckDB Tables

Results are written back to `hotel_reviews.db` so they can be queried alongside `REVIEW_DATA`.

### `TOPIC_LABELS` — master topic registry per model run

```sql
CREATE TABLE TOPIC_LABELS (
    run_id         VARCHAR NOT NULL,   -- e.g. "coast_band_A" or "year_2022"
    topic_id       INTEGER NOT NULL,   -- BERTopic topic number (-1 = outlier)
    top_words      VARCHAR,            -- comma-joined KeyBERT keywords
    seed_topic     VARCHAR,            -- assigned by LLM: service/location/clean/room/value
    seed_score     FLOAT,              -- LLM confidence (0–1)
    n_docs         INTEGER,            -- size of topic cluster
    PRIMARY KEY (run_id, topic_id)
);
```

### `REVIEW_TOPICS` — per-review topic assignment

```sql
CREATE TABLE REVIEW_TOPICS (
    run_id     VARCHAR  NOT NULL,
    review_id  VARCHAR  NOT NULL,
    topic_id   INTEGER  NOT NULL,
    prob       FLOAT,
    PRIMARY KEY (run_id, review_id)
);
```

These two tables replace ad-hoc `.pkl` files and make the line-graph queries trivial SQL joins.

---

## Seed Topics (LLM Labelling)

After each BERTopic fit, the top keywords for each discovered topic are sent to Claude with the five seed definitions. Claude assigns the closest seed label and a confidence score.

```python
seed_topics = {
    "service":     ["service", "reception", "staff", "support", "friendly",
                    "professional", "care", "help"],
    "location":    ["location", "center", "view", "convenient", "near",
                    "area", "transport", "accessible", "surroundings"],
    "cleanliness": ["clean", "tidy", "neat", "organized", "maintained", "orderly"],
    "room":        ["room", "amenities", "spacious", "comfortable", "bed",
                    "air conditioning"],
    "value":       ["value", "reasonable", "cost", "worth", "price", "quality",
                    "economical", "savings"],
}
```

Topics that don't resemble any seed (score < 0.3) stay labeled `"other"`. This keeps the labels honest — BERTopic may surface themes the five seeds don't cover (e.g. beach access, food, noise).

---

## Pipeline — Analysis 1: Distance-Band Models

Produces **3 independent checkpoints**, one per band.

```
run_id: "coast_band_A" | "coast_band_B" | "coast_band_C"
```

```python
# Pseudocode — src/run_coast_analysis.py

BANDS = {
    "coast_band_A": "r.distance2coastline < 0.1",
    "coast_band_B": "r.distance2coastline >= 0.1 AND r.distance2coastline < 0.5",
    "coast_band_C": "r.distance2coastline >= 0.5",
}

for run_id, where_clause in BANDS.items():
    ckpt = Path(f"checkpoints/{run_id}.pkl")
    if ckpt.exists():
        print(f"[{run_id}] checkpoint found — skipping fit")
        topic_model = BERTopic.load(str(ckpt))
    else:
        df, docs, embeddings = load_from_duckdb(extra_where=where_clause)
        topic_model = build_bertopic(min_cluster_size=30, min_topic_size=30)
        topics, probs = topic_model.fit_transform(docs, embeddings)
        topic_model.save(str(ckpt), serialization="pickle", save_ctfidf=True)
        _write_to_duckdb(run_id, df, topics, probs, topic_model)

    _label_topics_with_llm(run_id)   # writes seed_topic/seed_score to TOPIC_LABELS
```

Expected row counts per band (approximate — exact split depends on data):

| Band | Approx rows |
| --- | --- |
| Beachfront | ~30 k |
| Near-coast | ~90 k |
| Inland | ~130 k |

---

## Pipeline — Analysis 2: Year-Slice Models

Produces **7 independent checkpoints**, one per year 2018–2024.

```
run_id: "year_2018" … "year_2024"
```

```python
# Pseudocode — src/run_year_analysis.py

for year in range(2018, 2025):
    run_id = f"year_{year}"
    ckpt = Path(f"checkpoints/{run_id}.pkl")
    if ckpt.exists():
        print(f"[{run_id}] checkpoint found — skipping fit")
        topic_model = BERTopic.load(str(ckpt))
    else:
        df, docs, embeddings = load_from_duckdb(min_year=year, max_year=year)
        topic_model = build_bertopic(min_cluster_size=30, min_topic_size=30)
        topics, probs = topic_model.fit_transform(docs, embeddings)
        topic_model.save(str(ckpt), serialization="pickle", save_ctfidf=True)
        _write_to_duckdb(run_id, df, topics, probs, topic_model)

    _label_topics_with_llm(run_id)
```

Year coverage in the corpus:

| Year | Agoda reviews | GMaps reviews |
| --- | --- | --- |
| 2018 | ~8 k | ~3 k |
| 2019 | ~16 k | ~6 k |
| 2020 | ~10 k | ~4 k |
| 2021 | ~12 k | ~5 k |
| 2022 | ~18 k | ~8 k |
| 2023 | ~22 k | ~10 k |
| 2024 | ~15 k | ~7 k |

Years with < 5 k reviews may produce noisier topics; reduce `min_cluster_size` to 20 for those slices.

---

## LLM Labelling Step

After each `topic_model.fit_transform`, call Claude to label each topic. One API call per topic (batch if many topics share a model to reduce latency):

```python
# Pseudocode — src/llm_label.py

import anthropic

client = anthropic.Anthropic()

def label_topic(run_id: str, topic_id: int, top_words: list[str]) -> tuple[str, float]:
    """Return (seed_label, confidence)."""
    prompt = f"""
You are labelling hotel review topics for a Vietnamese coastal hotel study.

Seed topics and their keywords:
{json.dumps(seed_topics, ensure_ascii=False, indent=2)}

The discovered topic has these top words:
{", ".join(top_words)}

Pick the SINGLE closest seed topic label. If none fits, return "other".
Respond with JSON only: {{"label": "<seed_topic>", "confidence": <0.0-1.0>}}
"""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(response.content[0].text)
    return result["label"], result["confidence"]
```

Results written to `TOPIC_LABELS.seed_topic` and `TOPIC_LABELS.seed_score`.

---

## Visualisation — Line Graphs

Both charts use the same query pattern against `REVIEW_TOPICS` + `TOPIC_LABELS`.

### Chart 1 — Topic share over distance bands

X-axis: distance band (A → B → C).  
Y-axis: share of reviews assigned to each seed topic (%).  
One line per seed topic.

```sql
SELECT
    CASE
        WHEN r.distance2coastline < 0.1  THEN 'Beachfront'
        WHEN r.distance2coastline < 0.5  THEN 'Near-coast'
        ELSE 'Inland'
    END                            AS band,
    tl.seed_topic,
    COUNT(*)                       AS n_reviews,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (
        PARTITION BY band
    )                              AS pct
FROM REVIEW_TOPICS rt
JOIN REVIEW_LABELS  tl ON rt.run_id = tl.run_id AND rt.topic_id = tl.topic_id
JOIN REVIEW_DATA    r  ON rt.review_id = r.review_id
WHERE rt.run_id LIKE 'coast_band_%'
  AND rt.topic_id != -1
GROUP BY band, tl.seed_topic
ORDER BY band, tl.seed_topic;
```

### Chart 2 — Topic share over years (2018–2024)

X-axis: year.  
Y-axis: share of reviews per seed topic (%).  
One line per seed topic — allows you to see if "value" complaints rose post-COVID or "cleanliness" mentions spiked in a particular year.

```sql
SELECT
    CAST(rt.run_id AS VARCHAR)     AS year,   -- "year_2022" → parsed to 2022
    tl.seed_topic,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (
        PARTITION BY year
    )                              AS pct
FROM REVIEW_TOPICS rt
JOIN TOPIC_LABELS  tl ON rt.run_id = tl.run_id AND rt.topic_id = tl.topic_id
WHERE rt.run_id LIKE 'year_%'
  AND rt.topic_id != -1
GROUP BY year, tl.seed_topic
ORDER BY year, tl.seed_topic;
```

Both charts are rendered in `notebooks/11_topic_coast_time.ipynb` using **matplotlib** (line graph with markers, one colour per seed topic, shared legend).

---

## Checkpoint Strategy

```
checkpoints/
├── coast_band_A.pkl     # BERTopic for d < 0.1 km
├── coast_band_B.pkl     # BERTopic for 0.1–0.5 km
├── coast_band_C.pkl     # BERTopic for d ≥ 0.5 km
├── year_2018.pkl
├── year_2019.pkl
├── …
└── year_2024.pkl
```

Checkpoint existence is checked at the start of each loop iteration — safe to kill and resume at any point. `REVIEW_TOPICS` uses `INSERT OR IGNORE` so re-runs don't duplicate rows.

---

## Files Affected

| File | Action |
| --- | --- |
| `src/run_coast_analysis.py` | **New** — fits 3 distance-band models, writes checkpoints + DuckDB rows |
| `src/run_year_analysis.py` | **New** — fits 7 year-slice models, writes checkpoints + DuckDB rows |
| `src/llm_label.py` | **New** — calls Claude to assign seed topic per BERTopic topic |
| `src/topic_modeling.py` | Add `_write_to_duckdb()` helper (TOPIC_LABELS + REVIEW_TOPICS) |
| `notebooks/11_topic_coast_time.ipynb` | **New** — loads from DuckDB, draws both line graphs |
| `pyproject.toml` | Add `matplotlib` if not already present |

## What Does NOT Change

- `REVIEW_EMBEDDINGS` — same vectors, reused by both analyses
- `src/preprocessor.py`, `src/preprocess_to_duckdb.py`, `src/embed_to_duckdb.py` — untouched
- `build_bertopic()` config — same UMAP / HDBSCAN / KeyBERT setup, just `min_cluster_size` tuned per slice
- `HOTEL`, `AGODA_REVIEW`, `GOOGLEMAPS_REVIEW`, `REVIEW_DATA` — untouched

---

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Small band/year slices → noisy topics | Drop `min_cluster_size` to 20 for slices < 10 k rows; flag in chart annotation |
| Topic spaces differ per run — same seed can land on different BERTopic ID | LLM labelling normalises across runs via `seed_topic`; charts use seed label not raw topic_id |
| LLM rate limits on 10 sequential model runs | Batch all topic keywords per run into one API call; fall back to per-topic calls |
| `distance2coastline` nulls | Filter `WHERE distance2coastline IS NOT NULL` in band queries |
| Checkpoint `.pkl` bloat (~50–200 MB each × 10 runs) | `checkpoints/` is gitignored; regenerable from `REVIEW_EMBEDDINGS` |
