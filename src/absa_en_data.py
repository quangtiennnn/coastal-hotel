"""
absa_en_data.py
===============
Build the ABSA training set from the human-annotated Label Studio exports
(the gold TRAINING data — see ABSA_TRAINING_PROPOSAL.md).

Turns span-level annotations into **document-level multi-head labels**:
one review text -> a sentiment for each of the 6 macro aspects, where each
sentiment is one of {not_mentioned, negative, neutral, positive}.

Aspect taxonomy here is the annotators' native **6-class** set (includes
`Branding`) - NOT the 5-aspect `other`-folded scheme used for the LLM silver
labels. Model B is trained purely on this human gold.

Sources (Label Studio JSON; note the misleading filenames):
  tripadvisor    Annoted_TripAdvisor_EN.json        9,867 docs   English
  booking        Annotated Data/BOOKING_VN.json     6,977 docs   English (VN name)
  tripadvisor_vi Annotated Data/TripAdvisor_VN.json 17,624 docs  Vietnamese

Module keeps its EN name for import stability; it now handles both languages.
Used by: src/absa_train_en.py, notebooks/26_absa_train_en.ipynb (+ the bilingual notebook).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent

# 6 macro aspects, in a fixed order (= head order in the model)
ASPECTS = ["Facility", "Amenity", "Service", "Experience", "Loyalty", "Branding"]
# 4 sentiment classes per aspect, in a fixed order (= class index)
CLASSES = ["not_mentioned", "negative", "neutral", "positive"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
ID2CLASS = {i: c for c, i in CLASS2ID.items()}

_SENT = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}

_ANNOT = ROOT / "data" / "raw" / "Annotated Data"
EN_SOURCES = {  # kept name for import stability; now bilingual
    "tripadvisor": ROOT / "data" / "raw" / "Annoted_TripAdvisor_EN.json",  # en
    "booking": _ANNOT / "BOOKING_VN.json",                                  # en
    "tripadvisor_vi": _ANNOT / "TripAdvisor_VN.json",                       # vi
}
SOURCE_LANG = {"tripadvisor": "en", "booking": "en", "tripadvisor_vi": "vi"}


def doc_to_labels(item: dict) -> tuple[str, dict[str, str]]:
    """One Label Studio item -> (text, {aspect: sentiment}).

    A document may contain several spans of the same aspect; the document-level
    sentiment for that aspect is the MAJORITY vote over its spans, ties -> neutral
    (a genuinely mixed aspect, e.g. "room clean but bathroom dirty"). Aspects with
    no span are `not_mentioned`.
    """
    text = (item.get("data") or {}).get("text", "") or ""

    # Pair the aspect label and the sentiment label that share a span id.
    spans: dict[str, dict] = {}
    for ann in item.get("annotations", []):
        for res in ann.get("result", []):
            rid = res.get("id")
            labels = (res.get("value") or {}).get("labels") or []
            if rid is None or not labels:
                continue
            span = spans.setdefault(rid, {"aspect": None, "sentiment": None})
            if res.get("from_name") == "entities" and labels[0] in ASPECTS:
                span["aspect"] = labels[0]
            elif res.get("from_name") == "entity_sentiment":
                span["sentiment"] = _SENT.get(labels[0])

    per_aspect: dict[str, list[str]] = {a: [] for a in ASPECTS}
    for span in spans.values():
        if span["aspect"] and span["sentiment"]:
            per_aspect[span["aspect"]].append(span["sentiment"])

    labels: dict[str, str] = {}
    for aspect, sents in per_aspect.items():
        if not sents:
            labels[aspect] = "not_mentioned"
        else:
            counts = Counter(sents)
            best = max(counts.values())
            winners = [s for s, n in counts.items() if n == best]
            labels[aspect] = winners[0] if len(winners) == 1 else "neutral"
    return text, labels


def _is_annotated(item: dict) -> bool:
    """True only if a human actually annotated this item.

    Items with `annotations: []` were never labeled - counting them as
    all-`not_mentioned` would poison training with false negatives (a real
    issue for TripAdvisor_VN, where ~17% of items are unannotated).
    """
    for ann in item.get("annotations", []):
        if ann.get("result"):
            return True
    return False


def load_docs(
    sources: list[str] | None = None,
    annotated_only: bool = True,
    with_lang: bool = False,
) -> list:
    """Load + aggregate the chosen annotated sources.

    Returns [(text, [class_id per aspect]), ...] (with text non-empty), or
    [(text, labels, lang), ...] when `with_lang=True` (used to stratify the
    bilingual split by language).
    """
    sources = sources or ["tripadvisor"]
    rows: list = []
    for name in sources:
        lang = SOURCE_LANG[name]
        items = json.loads(EN_SOURCES[name].read_text(encoding="utf-8"))
        for item in items:
            if annotated_only and not _is_annotated(item):
                continue
            text, labels = doc_to_labels(item)
            if not text.strip():
                continue
            y = [CLASS2ID[labels[a]] for a in ASPECTS]
            rows.append((text, y, lang) if with_lang else (text, y))
    return rows


def split_docs(
    rows: list[tuple[str, list[int]]],
    val_frac: float = 0.1,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[str, list[int]]]]:
    """Deterministic random train/val/test split."""
    import random

    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = [rows[i] for i in idx[:n_test]]
    val = [rows[i] for i in idx[n_test:n_test + n_val]]
    train = [rows[i] for i in idx[n_test + n_val:]]
    return {"train": train, "val": val, "test": test}


def class_weights(train_rows: list[tuple[str, list[int]]]) -> list[list[float]]:
    """Per-aspect inverse-frequency class weights (for weighted cross-entropy).

    Reviews are ~2/3 positive and mostly not_mentioned per aspect; without this,
    the model just predicts the majority class and scores ~0 on neg/neutral.
    """
    weights: list[list[float]] = []
    for a in range(len(ASPECTS)):
        counts = Counter(y[a] for _, y in train_rows)
        total = sum(counts.values())
        w = []
        for c in range(len(CLASSES)):
            n = counts.get(c, 0)
            # inverse frequency, normalized so the mean weight ~ 1
            w.append(total / (len(CLASSES) * n) if n else 0.0)
        weights.append(w)
    return weights


def label_distribution(rows: list[tuple[str, list[int]]]) -> dict[str, Counter]:
    """{aspect: Counter(class_name -> n)} — for the notebook's data-inspection cell."""
    dist: dict[str, Counter] = {}
    for a, aspect in enumerate(ASPECTS):
        dist[aspect] = Counter(ID2CLASS[y[a]] for _, y in rows)
    return dist
