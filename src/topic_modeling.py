"""
topic_modeling.py
=================
BERTopic pipeline for hotel reviews stored in hotel_reviews.db.

Stages:
  1. Preprocessor      — normalize + word-segment per language (vi/en)
  2. EmbeddingEngine   — encode with paraphrase-multilingual-mpnet-base-v2
  3. load_from_duckdb()— load pre-computed embeddings from REVIEW_DATA
  4. build_bertopic()  — assemble BERTopic with UMAP + HDBSCAN + KeyBERT
  5. run_pipeline()    — end-to-end convenience wrapper
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from stopwordsiso import stopwords as _iso_stopwords

# BERTopic-stack imports are lazy (inside build_bertopic / run_pipeline) so that
# load_from_duckdb and Preprocessor remain importable even when the HDBSCAN DLL
# is unavailable (e.g. Windows AppControl restrictions).
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bertopic import BERTopic

DB_PATH = Path(__file__).parent.parent / "data" / "hotel_reviews.db"

EN_EXTRA_STOPWORDS: list[str] = [
    "hotel", "room", "stay", "stayed", "night", "check", "would", "also",
    "really", "got", "went", "said", "told", "time", "day", "place",
    "one", "two", "three", "us", "we", "our", "i", "was", "is", "are",
    "it", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "this", "that", "there", "were", "had",
    "have", "has", "be", "been", "will", "can", "could", "would", "should",
]


def build_stopwords() -> list[str]:
    return list(_iso_stopwords(["vi", "en"])) + EN_EXTRA_STOPWORDS


# ===========================================================================
# 1. Preprocessor
# ===========================================================================
class Preprocessor:
    """Normalize and word-segment each review by language.

    Vietnamese: ViTokenizer joins compound words with underscores.
    English:    spaCy tokenizer, whitespace-rejoined.
    """

    def __init__(self) -> None:
        from pyvi import ViTokenizer
        import spacy
        self._vi_tokenize = ViTokenizer.tokenize
        self._nlp = spacy.load("en_core_web_sm")
        print("[Preprocessor] Loaded ViTokenizer + spaCy en_core_web_sm.")

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text).lower())
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def segment(self, text: str, language: str) -> str:
        if language == "vi":
            return self._vi_tokenize(text)
        doc = self._nlp(text)
        return " ".join(t.text for t in doc if not t.is_space)

    def process(self, df: pd.DataFrame) -> list[str]:
        """Return processed_text list aligned to df rows.

        Expects columns: review_text, language.
        """
        print(f"[Preprocessor] Processing {len(df):,} rows …")
        texts = df["review_text"].astype(str).apply(self.normalize)
        processed = [
            self.segment(text, lang)
            for text, lang in zip(texts, df["language"])
        ]
        print("[Preprocessor] Done.")
        return processed


# ===========================================================================
# 2. EmbeddingEngine
# ===========================================================================
class EmbeddingEngine:
    """Encode texts with paraphrase-multilingual-mpnet-base-v2 (dim=768)."""

    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

    def __init__(self, batch_size: int = 64, show_progress: bool = True) -> None:
        self.batch_size = batch_size
        self.show_progress = show_progress
        print(f"[EmbeddingEngine] Loading model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        print(f"[EmbeddingEngine] Encoding {len(texts):,} texts …")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        print(f"[EmbeddingEngine] Done. Shape: {embeddings.shape}")
        return embeddings


# ===========================================================================
# 3. load_from_duckdb()
# ===========================================================================

def load_from_duckdb(
    db_path: str | Path = DB_PATH,
    language: Optional[str] = None,
    hotel_ids: Optional[list[int]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_rating: Optional[float] = None,
    extra_where: str = "",
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load pre-computed embeddings from REVIEW_DATA.

    Returns
    -------
    df         : DataFrame with all REVIEW_DATA columns (no embedding column)
    docs       : list[str] — processed_text aligned to df rows
    embeddings : float32 ndarray of shape (n, 768)

    Raises
    ------
    RuntimeError if embeddings are not yet computed (run embed_to_duckdb.py first).
    """
    # REVIEW_DATA is a view; processed text and embeddings live in separate tables.
    review_filters: list[str] = []
    if language:
        review_filters.append(f"r.language = '{language}'")
    if hotel_ids:
        ids = ", ".join(str(i) for i in hotel_ids)
        review_filters.append(f"r.hotel_id IN ({ids})")
    if min_year:
        review_filters.append(f"r.review_year >= {min_year}")
    if max_year:
        review_filters.append(f"r.review_year <= {max_year}")
    if min_rating:
        review_filters.append(f"r.rating_normalized >= {min_rating}")
    if extra_where:
        review_filters.append(f"({extra_where})")

    where_clause = ("AND " + " AND ".join(review_filters)) if review_filters else ""

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "REVIEW_TEXT_PROCESSED" not in tables or con.execute(
            "SELECT COUNT(*) FROM REVIEW_TEXT_PROCESSED"
        ).fetchone()[0] == 0:
            raise RuntimeError(
                "REVIEW_TEXT_PROCESSED is missing or empty. "
                "Run `uv run python src/preprocess_to_duckdb.py` first."
            )
        if "REVIEW_EMBEDDINGS" not in tables or con.execute(
            "SELECT COUNT(*) FROM REVIEW_EMBEDDINGS"
        ).fetchone()[0] == 0:
            raise RuntimeError(
                "REVIEW_EMBEDDINGS is missing or empty. "
                "Run `uv run python src/embed_to_duckdb.py` first."
            )

        count = con.execute(f"""
            SELECT COUNT(*)
            FROM REVIEW_DATA r
            INNER JOIN REVIEW_TEXT_PROCESSED p ON r.review_id = p.review_id
            INNER JOIN REVIEW_EMBEDDINGS e      ON r.review_id = e.review_id
            WHERE 1=1 {where_clause}
        """).fetchone()[0]

        if count == 0:
            raise RuntimeError("No rows match the given filters.")

        print(f"[load_from_duckdb] Loading {count:,} rows …")

        result = con.execute(f"""
            SELECT r.source, r.review_id, r.hotel_id, r.hotel_name, r.city,
                   r.star_rating, r.distance2coastline, r.review_text, r.language,
                   r.rating_normalized, r.review_year, r.review_month,
                   r.reviewer_nationality, r.reviewer_continent, r.is_local_guide,
                   r.asp_room, r.asp_service, r.asp_location, r.asp_food_drink,
                   r.asp_value, r.asp_cleanliness, r.label_model, r.labeled_at,
                   p.processed_text, e.embedding
            FROM REVIEW_DATA r
            INNER JOIN REVIEW_TEXT_PROCESSED p ON r.review_id = p.review_id
            INNER JOIN REVIEW_EMBEDDINGS e      ON r.review_id = e.review_id
            WHERE 1=1 {where_clause}
        """).df()
    finally:
        con.close()

    docs = result["processed_text"].tolist()
    embeddings = np.stack(result["embedding"].tolist()).astype("float32")
    df = result.drop(columns=["embedding"])

    print(f"[load_from_duckdb] docs: {len(docs):,}  embeddings: {embeddings.shape}")
    return df, docs, embeddings


# ===========================================================================
# 4. build_bertopic()
# ===========================================================================

def build_bertopic(
    nr_topics: str | int = "auto",
    min_cluster_size: int = 20,
    min_topic_size: int = 20,
    embedding_model: Optional[SentenceTransformer] = None,
) -> "BERTopic":
    """Assemble BERTopic with UMAP + HDBSCAN + KeyBERT + MMR representation.

    Notes
    -----
    - ``calculate_probabilities=False`` and ``prediction_data=False`` are fixed.
      Both flags load ``_prediction_utils.pyd`` on Windows, which is blocked by
      AppControl policies.  Hard topic assignments are unaffected.
    - Do NOT call ``topic_model.approximate_distribution()``,
      ``hdbscan.approximate_predict()``, or any method that reads the probability
      matrix — they all require the blocked DLL.
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        cluster_selection_method="eom",
        metric="euclidean",
        prediction_data=False,  # _prediction_utils.pyd — blocked by Windows AppControl
    )

    vectorizer_model = CountVectorizer(
        stop_words=build_stopwords(),
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    )

    ctfidf_model = ClassTfidfTransformer()

    representation_model = (
        {"KeyBERT": KeyBERTInspired(), "MMR": MaximalMarginalRelevance(diversity=0.3)}
        if embedding_model is not None
        else {"MMR": MaximalMarginalRelevance(diversity=0.3)}
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,  # _prediction_utils.pyd — blocked by Windows AppControl
        verbose=True,
    )

    print("[build_bertopic] BERTopic configured.")
    return topic_model


# ===========================================================================
# 5. run_pipeline()
# ===========================================================================

def run_pipeline(
    db_path: str | Path = DB_PATH,
    language: Optional[str] = None,
    hotel_ids: Optional[list[int]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_rating: Optional[float] = None,
    extra_where: str = "",
    nr_topics: str | int = "auto",
    min_cluster_size: int = 20,
    min_topic_size: int = 20,
) -> tuple["BERTopic", pd.DataFrame, list[int]]:
    """Load embeddings from DuckDB, fit BERTopic, return results.

    Returns
    -------
    topic_model : BERTopic
    df          : DataFrame (REVIEW_DATA rows, no embedding column)
    topics      : list[int] — hard topic label per document (-1 = outlier)

    Note: probabilities are not computed (AppControl blocks _prediction_utils.pyd).
    """
    engine = EmbeddingEngine()
    df, docs, embeddings = load_from_duckdb(
        db_path=db_path,
        language=language,
        hotel_ids=hotel_ids,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating,
        extra_where=extra_where,
    )

    topic_model = build_bertopic(
        nr_topics=nr_topics,
        min_cluster_size=min_cluster_size,
        min_topic_size=min_topic_size,
        embedding_model=engine.model,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    outliers = sum(1 for t in topics if t == -1)
    print(f"\n[run_pipeline] Done. Topics: {n_topics} | Outliers: {outliers:,}")

    return topic_model, df, topics
