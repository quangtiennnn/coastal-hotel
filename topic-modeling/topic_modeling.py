"""
topic-modeling.py
=================
BERTopic pipeline for Agoda hotel reviews.

Stages:
  1. Preprocessor      — normalize + word-segment per language (vi/en)
  2. ReviewLoader      — load agoda-reviews-en-vi.csv, apply Preprocessor
  3. EmbeddingEngine   — encode with paraphrase-multilingual-mpnet-base-v2
  4. QdrantStore       — persist vectors to Qdrant (Docker @ localhost:6333)
  5. build_bertopic()  — assemble BERTopic with representation models
  6. run_pipeline()    — end-to-end convenience wrapper
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords as _iso_stopwords
from umap import UMAP

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "agoda-reviews-en-vi.csv"

# ---------------------------------------------------------------------------
# Extra domain stopwords (hotel-specific noise on top of stopwordsiso)
# ---------------------------------------------------------------------------
EN_EXTRA_STOPWORDS: list[str] = [
    "hotel", "room", "stay", "stayed", "night", "check", "would", "also",
    "really", "got", "went", "said", "told", "time", "day", "place",
    "one", "two", "three", "us", "we", "our", "i", "was", "is", "are",
    "it", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "this", "that", "there", "were", "had",
    "have", "has", "be", "been", "will", "can", "could", "would", "should",
]


def build_stopwords() -> list[str]:
    """Combined vi + en stopwords from stopwordsiso plus domain extras."""
    return list(_iso_stopwords(["vi", "en"])) + EN_EXTRA_STOPWORDS


# ===========================================================================
# 1. Preprocessor
# ===========================================================================
class Preprocessor:
    """Normalize and word-segment each review by language before topic modeling.

    Vietnamese: ViTokenizer joins compound words with underscores
                e.g. 'khách sạn' → 'khách_sạn'
    English:    spaCy tokenizer, whitespace-rejoined
    """

    def __init__(self) -> None:
        from pyvi import ViTokenizer
        import spacy
        self._vi_tokenize = ViTokenizer.tokenize
        self._nlp = spacy.load("en_core_web_sm")
        print("[Preprocessor] Loaded ViTokenizer + spaCy en_core_web_sm.")

    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text).lower())
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    def segment(self, text: str, language: str) -> str:
        if language == "vi":
            return self._vi_tokenize(text)
        doc = self._nlp(text)
        return " ".join(t.text for t in doc if not t.is_space)

    # ------------------------------------------------------------------
    def process(self, df: pd.DataFrame) -> list[str]:
        """Return processed_comment list aligned to df rows."""
        print(f"[Preprocessor] Processing {len(df):,} rows …")
        texts = df["comment"].astype(str).apply(self.normalize)
        processed = [
            self.segment(text, lang)
            for text, lang in zip(texts, df["language"])
        ]
        print("[Preprocessor] Done.")
        return processed


# ===========================================================================
# 2. ReviewLoader
# ===========================================================================
class ReviewLoader:
    """Load agoda-reviews-en-vi.csv, clean, and return preprocessed texts."""

    def __init__(self, csv_path: str | Path = DEFAULT_CSV) -> None:
        self.csv_path = Path(csv_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig", low_memory=False)

        df = df.dropna(subset=["comment"])
        df = df[df["comment"].str.strip().astype(bool)].reset_index(drop=True)
        df["comment"] = df["comment"].str.strip()

        print(f"[ReviewLoader] Loaded {len(df):,} reviews with valid comments.")
        return df

    def get_texts(self, preprocess: bool = True) -> tuple[pd.DataFrame, list[str]]:
        df = self.load()
        if preprocess:
            texts = Preprocessor().process(df)
        else:
            texts = df["comment"].tolist()
        return df, texts


# ===========================================================================
# 3. EmbeddingEngine
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
# 4. QdrantStore
# ===========================================================================
COLLECTION_NAME = "agoda_reviews_envi"
VECTOR_DIM = 768


class QdrantStore:
    """Qdrant vector store backed by Docker container at localhost:6333."""

    def __init__(self, host: str = "localhost", port: int = 6333, collection: str = COLLECTION_NAME) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        print(f"[QdrantStore] Connected to Qdrant at {host}:{port}  collection='{collection}'")

    # ------------------------------------------------------------------
    def create_collection(self, recreate: bool = False) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            if recreate:
                self.client.delete_collection(self.collection)
                print(f"[QdrantStore] Deleted existing collection '{self.collection}'.")
            else:
                print(f"[QdrantStore] Collection '{self.collection}' already exists — skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"[QdrantStore] Created collection '{self.collection}' (dim={VECTOR_DIM}).")

    # ------------------------------------------------------------------
    def upsert(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        batch_size: int = 256,
    ) -> None:
        """Upsert vectors with the full DataFrame row as payload."""
        n = len(df)
        print(f"[QdrantStore] Upserting {n:,} points in batches of {batch_size} …")

        records = df.to_dict(orient="records")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{start + i}")),
                    vector=embeddings[start + i].tolist(),
                    payload={
                        k: (None if pd.isna(v) else v)
                        for k, v in records[start + i].items()
                    },
                )
                for i in range(end - start)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

        print(f"[QdrantStore] Upsert complete. Total points: {n:,}")

    # ------------------------------------------------------------------
    def fetch_all(self) -> tuple[list[str], np.ndarray]:
        """Return (docs, embeddings) by scrolling the entire collection."""
        print("[QdrantStore] Fetching all vectors from Qdrant …")

        all_points = []
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_vectors=True,
                with_payload=True,
                limit=1000,
                offset=offset,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        docs = [p.payload.get("comment", "") for p in all_points]
        embeddings = np.array([p.vector for p in all_points], dtype=np.float32)

        print(f"[QdrantStore] Fetched {len(docs):,} points. Embedding shape: {embeddings.shape}")
        return docs, embeddings

    # ------------------------------------------------------------------
    def collection_count(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.points_count


# ===========================================================================
# 5. build_bertopic()
# ===========================================================================
def build_bertopic(
    nr_topics: str | int = "auto",
    min_cluster_size: int = 50,
    min_topic_size: int = 50,
    embedding_model: Optional[SentenceTransformer] = None,
) -> BERTopic:
    """
    Assemble BERTopic with representation models from process...ipynb.

    Parameters
    ----------
    nr_topics : 'auto' or int
        Number of topics after automatic merging.
    min_cluster_size : int
        HDBSCAN minimum cluster size. Use 5–10 for the Vietnamese subset
        (~139 docs), 50 for the full en/vi dataset.
    min_topic_size : int
        BERTopic minimum topic size. Should match min_cluster_size.
    embedding_model : SentenceTransformer, optional
        Required by KeyBERTInspired for keyword re-ranking. Pass the same
        model used to compute the pre-computed embeddings. Pre-computed
        embeddings are still used for UMAP/HDBSCAN — the model is only
        called by the KeyBERT representation step.
    """
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_method="eom",
        metric="euclidean",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=build_stopwords(),
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    )

    ctfidf_model = ClassTfidfTransformer()

    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR":     MaximalMarginalRelevance(diversity=0.3),
    } if embedding_model is not None else {
        "MMR":     MaximalMarginalRelevance(diversity=0.3),
    }

    topic_model = BERTopic(
        embedding_model=embedding_model,  # needed by KeyBERT; pre-computed embeddings still used for clustering
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    print("[build_bertopic] BERTopic configured.")
    return topic_model


# ===========================================================================
# 6. run_pipeline()  — end-to-end convenience wrapper
# ===========================================================================
def run_pipeline(
    csv_path: str | Path = DEFAULT_CSV,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = COLLECTION_NAME,
    recreate_collection: bool = False,
    skip_embed_if_exists: bool = True,
    nr_topics: str | int = "auto",
    min_cluster_size: int = 50,
    min_topic_size: int = 50,
) -> tuple[BERTopic, list[int], np.ndarray]:
    """
    Full pipeline: CSV → preprocess → embed → Qdrant → BERTopic.

    Returns
    -------
    model  : BERTopic
    topics : list[int]  — topic label per document (-1 = outlier)
    probs  : np.ndarray — probability matrix (n_docs, n_topics)
    """
    store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=collection)
    store.create_collection(recreate=recreate_collection)

    if skip_embed_if_exists and store.collection_count() > 0:
        print("[run_pipeline] Collection already has data — skipping embed phase.")
        docs, embeddings = store.fetch_all()
    else:
        loader = ReviewLoader(csv_path)
        df, texts = loader.get_texts(preprocess=True)

        engine = EmbeddingEngine()
        embeddings = engine.encode(texts)

        store.upsert(df, embeddings)
        docs = texts

    topic_model = build_bertopic(
        nr_topics=nr_topics,
        min_cluster_size=min_cluster_size,
        min_topic_size=min_topic_size,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    outliers = sum(1 for t in topics if t == -1)
    print(f"\n[run_pipeline] Done. Topics found: {n_topics} | Outliers: {outliers:,}")

    return topic_model, topics, probs
