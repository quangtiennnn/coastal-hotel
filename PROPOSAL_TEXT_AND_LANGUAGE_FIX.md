# Proposal: Fix Text Pipeline & Split BERTopic by Language

## What Is Currently Broken

### Bug 1 — `docs` is not cast to `str`

`load_from_duckdb()` returns:

```python
docs = result["processed_text"].tolist()   # ← plain Python list, no astype(str)
```

If any row has a `NULL` in `REVIEW_TEXT_PROCESSED.processed_text`, it comes through as a Python `float('nan')`.  The `CountVectorizer` internally calls `str()` on each document, so `nan` becomes the 4-character string `"nan"`, which gets tokenised and pollutes every topic's keywords.

**Fix:** one line in `load_from_duckdb`:

```python
docs = result["processed_text"].fillna("").astype(str).tolist()
```

---

### Bug 2 — `REVIEW_EMBEDDINGS` were computed on `processed_text`, but the embedding layer must match `docs`

`embed_to_duckdb.py` reads `processed_text` from `REVIEW_TEXT_PROCESSED` and encodes it.  `load_from_duckdb` returns those same stored vectors.  So the embedding already corresponds to `processed_text`.

**This is correct and should not change.** Both `docs` (for the c-TF-IDF vectoriser) and `embeddings` (for UMAP/HDBSCAN) must come from the same field — `processed_text`.  The current code is architecturally right; it just needs the `astype(str)` guard above.

---

### Bug 3 — Vietnamese and English reviews are mixed in one BERTopic run

Mixing languages in a single BERTopic call means the `CountVectorizer` vocabulary spans both languages simultaneously.  The c-TF-IDF score for a Vietnamese-dominant cluster ends up with fragmented keywords like:

```
tin, gic, hpl, gitr, phhp, chtlng, phhp tin, gic hpl, price, sch
```

These are Vietnamese words whose diacritics were dropped mid-pipeline (encoding mismatch at the vectoriser level when the corpus is mixed).  A Vietnamese-only corpus passed to a CountVectorizer with Vietnamese stopwords produces readable output like the 13-example notebook:

```
khách_sạn, sạch_sẽ, phòng_ốc, tiện_nghi, nhân_viên, giá_cả, phù_hợp
```

**Fix:** run BERTopic once for `language="en"` and once for `language="vi"` per slice.

---

### Bug 4 — CountVectorizer is language-agnostic

The current `build_bertopic()` uses `build_stopwords()` which merges both `vi` and `en` stopword lists.  This is wasteful and can suppress legitimate en keywords in a vi corpus and vice versa.

**Fix:** pass the active language to `build_bertopic` and use only that language's stopwords in the vectoriser.

---

## New Run-ID Convention

Each slice is now split by language, giving **20 checkpoints** total:

| Run ID | Slice | Language |
| --- | --- | --- |
| `coast_band_A_en` | `distance2coastline < 0.1` | English |
| `coast_band_A_vi` | `distance2coastline < 0.1` | Vietnamese |
| `coast_band_B_en` | `0.1 ≤ distance2coastline < 0.5` | English |
| `coast_band_B_vi` | `0.1 ≤ distance2coastline < 0.5` | Vietnamese |
| `coast_band_C_en` | `distance2coastline ≥ 0.5` | English |
| `coast_band_C_vi` | `distance2coastline ≥ 0.5` | Vietnamese |
| `year_2018_en` … `year_2024_en` | per year | English |
| `year_2018_vi` … `year_2024_vi` | per year | Vietnamese |

---

## Changes to `src/topic_modeling.py`

### `load_from_duckdb` — fix `docs` casting

```python
# Before
docs = result["processed_text"].tolist()

# After
docs = result["processed_text"].fillna("").astype(str).tolist()
```

No other change to this function.

### `build_bertopic` — add `language` parameter

```python
def build_bertopic(
    nr_topics: str | int = "auto",
    min_cluster_size: int = 20,
    min_topic_size: int = 20,
    embedding_model=None,
    language: str = "en",          # ← new: "en" or "vi"
) -> BERTopic:
```

Inside, the stopword list and `CountVectorizer` token pattern change per language:

```python
# English — keep current token_pattern
if language == "en":
    stop_words = list(_iso_stopwords(["en"])) + EN_EXTRA_STOPWORDS
    token_pattern = r"(?u)\b\w\w+\b"

# Vietnamese — underscore-joined compound words like khách_sạn must not be split
elif language == "vi":
    stop_words = list(_iso_stopwords(["vi"]))
    token_pattern = r"[\w_][\w_]+"   # allows underscores inside tokens
```

Both go into `CountVectorizer(stop_words=..., token_pattern=..., min_df=2, ngram_range=(1,2))`.

---

## Changes to `notebooks/11_topic_coast_time.ipynb`

### Section 1 — Config: two languages, 20 runs

```python
LANGUAGES = ["en", "vi"]

BAND_SLICES = {
    "coast_band_A": "r.distance2coastline < 0.1",
    "coast_band_B": "r.distance2coastline >= 0.1 AND r.distance2coastline < 0.5",
    "coast_band_C": "r.distance2coastline >= 0.5",
}

YEAR_SLICES = {f"year_{y}": y for y in range(2018, 2025)}
```

### Section 4 (bands) and Section 5 (years) — inner loop over language

```python
for run_id_base, where_clause in BAND_SLICES.items():
    for lang in LANGUAGES:
        run_id = f"{run_id_base}_{lang}"
        ...
        df, docs, embeddings = load_from_duckdb(
            db_path=DB_PATH,
            language=lang,                          # ← language filter
            extra_where=f"{where_clause} AND r.distance2coastline IS NOT NULL",
        )
        # docs already str-safe from the load_from_duckdb fix
        topic_model, topics = fit_and_save(run_id, docs, embeddings, min_cs, ckpt, lang)
```

### `fit_and_save` — forward `language` to `build_bertopic`

```python
def fit_and_save(run_id, docs, embeddings, min_cs, ckpt, language="en"):
    topic_model = build_bertopic(
        nr_topics="auto",
        min_cluster_size=min_cs,
        min_topic_size=min_cs,
        language=language,          # ← passes correct stopwords + token_pattern
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    ...
```

---

## Files Affected

| File | Change |
| --- | --- |
| `src/topic_modeling.py` | `load_from_duckdb`: add `.fillna("").astype(str)` to `docs` |
| `src/topic_modeling.py` | `build_bertopic`: add `language` param, split stopwords + token_pattern per language |
| `notebooks/11_topic_coast_time.ipynb` | Section 1: add `LANGUAGES` list |
| `notebooks/11_topic_coast_time.ipynb` | Section 3 (`fit_and_save`): accept + forward `language` |
| `notebooks/11_topic_coast_time.ipynb` | Sections 4 & 5: add inner `for lang in LANGUAGES` loop |

## What Does NOT Change

- `REVIEW_EMBEDDINGS` — embeddings stay on `processed_text`, unchanged
- `embed_to_duckdb.py` — no change; multilingual model covers both languages
- `preprocess_to_duckdb.py` — no change; ViTokenizer for vi, spaCy for en already correct
- HDBSCAN params (`min_cluster_size=20`, `min_samples=5`, `prediction_data=False`)
- BERTopic params (`calculate_probabilities=False`, `nr_topics="auto"`)
- Checkpoint structure — just 20 files instead of 10

## Expected Outcome

| Language | Example topic keywords (before) | After fix |
| --- | --- | --- |
| Vietnamese | `tin, gic, hpl, gitr, phhp` | `khách_sạn, sạch_sẽ, nhân_viên, giá_cả, phù_hợp` |
| English | `hotel, room, staff, clean, location` | unchanged (was already correct when isolated) |
