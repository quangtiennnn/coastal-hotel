"""
preprocessor.py
===============
Standalone text preprocessor for the merged hotel review dataset.

Adapted from topic-modeling/topic_modeling.py — uses df["review_text"] and
df["language"] instead of df["comment"] / df["language"].

Usage
-----
    from preprocessor import Preprocessor

    pre = Preprocessor()
    df  = pre.process(df)           # adds 'processed_text' column in-place
    # or
    texts = pre.process_texts(df)   # returns list[str] only
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


class Preprocessor:
    """Normalize and word-segment each review by language.

    Vietnamese : ViTokenizer joins compound words with underscores
                 e.g. 'khách sạn' → 'khách_sạn'
    English    : spaCy tokenizer, whitespace-rejoined
    """

    def __init__(self) -> None:
        from pyvi import ViTokenizer
        import spacy

        self._vi_tokenize = ViTokenizer.tokenize
        self._nlp = spacy.load("en_core_web_sm")
        print("[Preprocessor] Loaded ViTokenizer + spaCy en_core_web_sm.")

    # ------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        """Lowercase, NFC unicode, strip punctuation, collapse whitespace."""
        text = unicodedata.normalize("NFC", str(text).lower())
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    def segment(self, text: str, language: str) -> str:
        """Word-segment text according to its language."""
        if language == "vi":
            return self._vi_tokenize(text)
        doc = self._nlp(text)
        return " ".join(t.text for t in doc if not t.is_space)

    # ------------------------------------------------------------------
    def process_texts(self, df: pd.DataFrame) -> list[str]:
        """Return a processed text list aligned to df rows.

        Reads df["review_text"] and df["language"].
        """
        print(f"[Preprocessor] Processing {len(df):,} rows …")
        normalized = df["review_text"].astype(str).apply(self.normalize)
        processed = [
            self.segment(text, lang)
            for text, lang in zip(normalized, df["language"])
        ]
        print("[Preprocessor] Done.")
        return processed

    # ------------------------------------------------------------------
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'processed_text' column to df and return it."""
        df = df.copy()
        df["processed_text"] = self.process_texts(df)
        return df
