"""
absa_model.py
=============
Model B's architecture, its persistence format, and the two aggregation rules -
shared by src/absa_train_b.py and src/absa_eval.py so that training and scoring
can never drift apart.

Architecture (ABSA_TRAINING_PROPOSAL.md, "Model B - architecture"):
joint ABSA as **multi-head classification**, not span extraction (there is no
in-domain span gold, and vi token alignment is fragile):

    shared multilingual encoder (xlm-roberta-base)
      -> 5 parallel 4-class heads, one per macro aspect
         (not_mentioned / negative / neutral / positive)
    joint loss = sum of class-weighted cross-entropies

Aggregation: Model B is trained on SENTENCES but scored against DOCUMENT-level
human annotations, so sentence predictions are pooled per review. Both the
primary rule and the pre-specified Rule B live here (see `aggregate_review`).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from absa_validate import ASPECTS5, CLASS2ID, CLASSES, ID2CLASS  # noqa: F401

ROOT = Path(__file__).parent.parent
MODEL_ROOT = ROOT / "models" / "absa_b"
DEFAULT_ENCODER = "xlm-roberta-base"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiAspectClassifier(nn.Module):
    """Shared encoder + one independent linear head per macro aspect."""

    def __init__(self, encoder_name: str = DEFAULT_ENCODER,
                 n_aspects: int = len(ASPECTS5),
                 n_classes: int = len(CLASSES), dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, n_classes) for _ in range(n_aspects)]
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])  # <s> / [CLS]
        return [head(cls) for head in self.heads]        # list of (B, n_classes)


class SentenceDataset(Dataset):
    """[(text, [class_id per aspect]), ...] -> (text, labels) pairs.

    Tokenization happens in the collate function, not here, so each batch is
    padded to ITS OWN longest sentence instead of to max_len. Review sentences
    are short - padding all of them to 128 tokens would spend most of the
    compute on padding, which matters when the ablation table needs four
    training runs.
    """

    def __init__(self, rows, tokenizer=None, max_len: int = 128):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i][0], self.rows[i][1]


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(texts, truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], labels
    return collate


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, arm: str, meta: dict) -> Path:
    out = MODEL_ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tokenizer.save_pretrained(out)
    (out / "config.json").write_text(
        json.dumps({"aspects": ASPECTS5, "classes": CLASSES, **meta},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def load_model(arm: str, device=None):
    """(model.eval(), tokenizer, config) for a trained ablation arm."""
    path = MODEL_ROOT / arm
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise SystemExit(
            f"No trained model at {path}. Train it first:\n"
            f"  uv run python src/absa_train_b.py --arm {arm}"
        )
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiAspectClassifier(cfg["encoder"], len(cfg["aspects"]),
                                  len(cfg["classes"]))
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(str(path))
    return model, tok, cfg


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_sentences(model, tokenizer, sentences: list[str], device,
                      batch_size: int = 64, max_len: int = 128
                      ) -> list[dict[str, str]]:
    """[sentence, ...] -> [{aspect: class_name}, ...] in the same order."""
    out: list[dict[str, str]] = []
    model.eval()
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        logits = model(enc["input_ids"], enc["attention_mask"])
        preds = [lg.argmax(1).cpu().tolist() for lg in logits]
        for j in range(len(chunk)):
            out.append({a: ID2CLASS[preds[k][j]] for k, a in enumerate(ASPECTS5)})
    return out


# ---------------------------------------------------------------------------
# Sentence -> review aggregation (proposal, "Sentence-to-Review Aggregation")
# ---------------------------------------------------------------------------

def aggregate_review(sentence_predictions: dict[str, list[str]], aspect: str,
                     rule: str = "primary") -> str:
    """Pool one review's per-sentence predictions for one aspect.

    `sentence_predictions[aspect]` is the list of that aspect's per-sentence
    labels for a single review.

    primary  An aspect is PRESENT if any sentence predicts a non-not_mentioned
             label; its sentiment is the MAJORITY over those sentences, with
             ties broken toward negative, then positive.

             Ties break toward negative on purpose: in a hotel-management
             context the two error types are not symmetric - a missed complaint
             means an operational problem goes unsurfaced, while a false
             positive costs only a wasted look at a non-issue.

    ruleB    "any negative sentence => review negative", regardless of
             majority. The pre-specified sensitivity check (ablation row 5);
             it is scored, never selected between.
    """
    relevant = [p for p in sentence_predictions.get(aspect, [])
                if p != "not_mentioned"]
    if not relevant:
        return "not_mentioned"

    if rule == "ruleB":
        return "negative" if "negative" in relevant else _majority(relevant)
    if rule != "primary":
        raise ValueError(f"unknown aggregation rule: {rule}")
    return _majority(relevant)


def _majority(relevant: list[str]) -> str:
    counts = Counter(relevant)
    max_count = max(counts.values())
    winners = [label for label, c in counts.items() if c == max_count]
    if len(winners) > 1:
        if "negative" in winners:
            return "negative"
        if "positive" in winners:
            return "positive"
        return "neutral"
    return winners[0]


AGGREGATION_RULES = ["primary", "ruleB"]


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def class_weights(train_rows) -> list[list[float]]:
    """Per-aspect inverse-frequency weights for weighted cross-entropy.

    Sentences are overwhelmingly not_mentioned for any single aspect and ~2/3
    positive when mentioned; the prior intuition is that unweighted training
    collapses onto the majority class. That intuition is NOT assumed here - it
    is ablation row 3 (weights on vs off) in Model B's own setup.
    """
    weights: list[list[float]] = []
    for a in range(len(ASPECTS5)):
        counts = Counter(row[1][a] for row in train_rows)
        total = sum(counts.values())
        weights.append([
            total / (len(CLASSES) * counts[c]) if counts.get(c) else 0.0
            for c in range(len(CLASSES))
        ])
    return weights
