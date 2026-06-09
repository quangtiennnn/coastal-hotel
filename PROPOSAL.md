# Proposal: Fix Topic-Implementation Pipeline with DuckDB

## Current State & Problems

The existing pipeline has two separate, fragile paths:

| Old path | Problem |
| --- | --- |
| `embeddings_{lang}.npy` flat files | Detached from data — index alignment breaks silently when rows are filtered/reordered |
| Qdrant Docker container | External service dependency; requires `docker run` before any notebook; over-engineered for this use case |
| Two CSVs (`final-reviews-en.csv`, `final-reviews-vi.csv`) | Duplicates data that now lives in `hotel_reviews.db` |

`REVIEW_DATA` in `hotel_reviews.db` already has **251,328 unified rows** from both Agoda and Google Maps. This is the single source of truth — processed text and embeddings should live alongside it, linked by `review_id`.

> **Note:** `REVIEW_DATA` is a `UNION ALL` view over `AGODA_REVIEW` and `GOOGLEMAPS_REVIEW`. Columns cannot be added to it directly — new data lives in separate tables joined on `review_id`.

---

## DuckDB vs Qdrant — Is the Switch Worth It?

**Short answer: yes, for this project.**

Qdrant's core value is **approximate nearest-neighbour (ANN) search** — finding the top-K most similar vectors in milliseconds at scale. BERTopic never does that. It loads **all** embeddings into a numpy array, runs UMAP, then HDBSCAN. The workflow is:

```text
SELECT embedding FROM REVIEW_EMBEDDINGS  →  np.array(...)  →  UMAP  →  HDBSCAN  →  topics
```

Qdrant adds Docker overhead, a network socket, UUID generation, and scroll-pagination — for a job that just needs `np.load()` semantics. DuckDB stores `FLOAT[768]` natively, lives in the same `hotel_reviews.db` file already tracked by the project, and returns a numpy array in one query.

**Verdict: DuckDB wins for this pipeline. Qdrant is unnecessary complexity here.**

---

## Two-Table Design

Preprocessing (word segmentation) and embedding are separate, expensive steps. Splitting them into two tables means:

- You can reuse `REVIEW_TEXT_PROCESSED` across multiple embedding models without re-running spaCy/ViTokenizer
- You can re-embed from `REVIEW_TEXT_PROCESSED` without re-running preprocessing
- Each step is independently resumable

### Table 1 — `REVIEW_TEXT_PROCESSED`

Stores the output of `src/preprocessor.py` (normalize → ViTokenizer / spaCy word-segmentation).

```sql
CREATE TABLE REVIEW_TEXT_PROCESSED (
    review_id      VARCHAR PRIMARY KEY,   -- FK → REVIEW_DATA.review_id
    processed_text VARCHAR NOT NULL
);
```

**Populated by:** `src/preprocess_to_duckdb.py`

### Table 2 — `REVIEW_EMBEDDINGS`

Stores the 768-dim `paraphrase-multilingual-mpnet-base-v2` vector for each review.

```sql
CREATE TABLE REVIEW_EMBEDDINGS (
    review_id VARCHAR PRIMARY KEY,        -- FK → REVIEW_TEXT_PROCESSED.review_id
    embedding FLOAT[768] NOT NULL
);
```

**Populated by:** `src/embed_to_duckdb.py` — reads `processed_text` from `REVIEW_TEXT_PROCESSED`

---

## Step 1 — Preprocessing (`preprocess_to_duckdb.py`)

Uses `src/preprocessor.Preprocessor` directly — no duplication.

```python
# Pseudocode

from src.preprocessor import Preprocessor

def preprocess_to_duckdb(db_path, batch_size=10_000):
    con = duckdb.connect(db_path)
    _ensure_table(con)          # CREATE TABLE IF NOT EXISTS REVIEW_TEXT_PROCESSED

    preprocessor = Preprocessor()

    while True:
        batch_df = con.execute("""
            SELECT r.review_id, r.review_text, r.language
            FROM REVIEW_DATA r
            WHERE NOT EXISTS (
                SELECT 1 FROM REVIEW_TEXT_PROCESSED p WHERE p.review_id = r.review_id
            )
            AND TRIM(COALESCE(r.review_text, '')) != ''
            LIMIT ?
        """, [batch_size]).df()

        if batch_df.empty:
            break

        processed = preprocessor.process_texts(batch_df)   # list[str]

        con.executemany(
            "INSERT OR IGNORE INTO REVIEW_TEXT_PROCESSED VALUES (?, ?)",
            list(zip(batch_df["review_id"], processed))
        )
```

### Step 1 — Expected Runtime

| Milestone | Rows completed | Approx time (CPU) |
| --- | --- | --- |
| Batch 1 | 10,000 | ~15 sec |
| … | … | … |
| Batch 26 (final) | 251,328 | ~6 min |

---

## Step 2 — Embedding (`embed_to_duckdb.py`)

Reads `processed_text` from `REVIEW_TEXT_PROCESSED` — does not re-run spaCy/ViTokenizer.

```python
# Pseudocode

from sentence_transformers import SentenceTransformer

def embed_to_duckdb(db_path, batch_size=10_000):
    con = duckdb.connect(db_path)
    _ensure_table(con)          # CREATE TABLE IF NOT EXISTS REVIEW_EMBEDDINGS

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    while True:
        batch_df = con.execute("""
            SELECT p.review_id, p.processed_text
            FROM REVIEW_TEXT_PROCESSED p
            WHERE NOT EXISTS (
                SELECT 1 FROM REVIEW_EMBEDDINGS e WHERE e.review_id = p.review_id
            )
            LIMIT ?
        """, [batch_size]).df()

        if batch_df.empty:
            break

        embeddings = model.encode(
            batch_df["processed_text"].tolist(),
            batch_size=64,
            show_progress_bar=True,
        ).astype("float32")

        con.executemany(
            "INSERT OR IGNORE INTO REVIEW_EMBEDDINGS VALUES (?, ?)",
            [(rid, emb.tolist()) for rid, emb in zip(batch_df["review_id"], embeddings)]
        )
```

### Step 2 — Expected Runtime

| Milestone | Rows completed | Approx time (CPU) |
| --- | --- | --- |
| Batch 1 | 10,000 | ~1.5 min |
| … | … | … |
| Batch 26 (final) | 251,328 | ~38 min |

Both steps are idempotent — interrupt and resume at any time.

---

## How BERTopic Loads Data

After both steps, `load_from_duckdb()` in `src/topic_modeling.py` JOINs all three:

```sql
SELECT r.*, p.processed_text, e.embedding
FROM REVIEW_DATA r
INNER JOIN REVIEW_TEXT_PROCESSED p ON r.review_id = p.review_id
INNER JOIN REVIEW_EMBEDDINGS e     ON r.review_id = e.review_id
WHERE r.language = 'en'
  AND r.review_year >= 2023
```

Returns `(df, docs, embeddings)` — no `.npy` files, no Qdrant, no index alignment bugs.

---

## Files Affected

| File | Action |
| --- | --- |
| `src/preprocessor.py` | Unchanged — used directly by preprocess_to_duckdb |
| `src/preprocess_to_duckdb.py` | **New** — batch preprocessor → `REVIEW_TEXT_PROCESSED` |
| `src/embed_to_duckdb.py` | **Updated** — reads from `REVIEW_TEXT_PROCESSED`, writes to `REVIEW_EMBEDDINGS` |
| `src/topic_modeling.py` | Remove `QdrantStore`, `ReviewLoader`; `load_from_duckdb()` uses the two new tables |
| `notebooks/09_topic_implement.ipynb` | Rewrite Sections 3–4 to load from DuckDB |
| `notebooks/10_topic_process.ipynb` | Update `run()` to use `load_from_duckdb()` |
| `pyproject.toml` | Add `duckdb`; remove `qdrant-client` |

## What Does NOT Change

- `paraphrase-multilingual-mpnet-base-v2` — same encoder
- `src/preprocessor.Preprocessor` — same class, just called from a script instead of a notebook
- BERTopic config (UMAP, HDBSCAN, KeyBERT, MMR) — same `build_bertopic()`
- `HOTEL`, `AGODA_REVIEW`, `GOOGLEMAPS_REVIEW` tables — untouched
- `REVIEW_DATA` view — untouched

---

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| `FLOAT[768]` inflates `.db` file (~750 MB for 251k rows) | Acceptable; DuckDB compresses arrays. Still smaller than a Qdrant Docker volume. |
| Crash mid-run | `NOT EXISTS` checkpoint — restart resumes from last complete row |
| Step 2 run before Step 1 | `embed_to_duckdb.py` checks `REVIEW_TEXT_PROCESSED` row count and raises early with a clear message |
| `FLOAT[]` → numpy round-trip precision | `float32` preserved end-to-end |
