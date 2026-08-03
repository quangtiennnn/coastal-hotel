"""
absa_train_en.py
================
Train the Stage-1 English ABSA model: a shared transformer encoder + 6 parallel
4-class heads (one macro aspect each), joint loss. Monolingual English
(distilbert-base-uncased) - the bilingual XLM-R version comes later once the
Vietnamese annotations are added.

This is the headless / reproducible companion to
notebooks/26_absa_train_en.ipynb (which explains every step). Same logic; the
notebook is for understanding, this script is for running.

Usage:
    # quick CPU run (subset, 2 epochs) - finishes in a few minutes
    uv run python src/absa_train_en.py --max-train 3000 --epochs 2

    # full run (all TripAdvisor docs, 3 epochs) - slow on CPU (~1h), fast on GPU
    uv run python src/absa_train_en.py --epochs 3

    # add the Booking source too
    uv run python src/absa_train_en.py --sources tripadvisor booking

Saves the fine-tuned model + tokenizer + label config to models/absa_en/.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

from absa_en_data import (
    ASPECTS,
    CLASSES,
    ID2CLASS,
    class_weights,
    load_docs,
    split_docs,
)

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "absa_en"
DEFAULT_ENCODER = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Dataset: text -> (input_ids, attention_mask, 6 label ints)
# ---------------------------------------------------------------------------

class AspectDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, labels = self.rows[i]
        enc = self.tok(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return (
            enc["input_ids"][0],
            enc["attention_mask"][0],
            torch.tensor(labels, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model: shared encoder + one linear head per aspect
# ---------------------------------------------------------------------------

class MultiAspectClassifier(nn.Module):
    def __init__(self, encoder_name: str, n_aspects: int, n_classes: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        # One independent classifier per aspect, all reading the same [CLS] vector.
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, n_classes) for _ in range(n_aspects)]
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])  # [CLS] token
        return [head(cls) for head in self.heads]        # list of (B, n_classes)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    preds = {a: [] for a in range(len(ASPECTS))}
    gold = {a: [] for a in range(len(ASPECTS))}
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(ids, mask)
            for a in range(len(ASPECTS)):
                preds[a] += logits[a].argmax(1).cpu().tolist()
                gold[a] += y[:, a].tolist()
    per_aspect = {}
    for a, aspect in enumerate(ASPECTS):
        per_aspect[aspect] = f1_score(
            gold[a], preds[a], average="macro", zero_division=0
        )
    macro = sum(per_aspect.values()) / len(per_aspect)
    return macro, {"per_aspect_f1": per_aspect, "preds": preds, "gold": gold}


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  encoder: {args.encoder}")

    rows = load_docs(args.sources)
    splits = split_docs(rows)
    if args.max_train:
        splits["train"] = splits["train"][: args.max_train]
    print(f"Docs: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}  (sources={args.sources})")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    make = lambda rows, shuffle: DataLoader(
        AspectDataset(rows, tok, args.max_len),
        batch_size=args.batch, shuffle=shuffle,
    )
    train_dl = make(splits["train"], True)
    val_dl = make(splits["val"], False)
    test_dl = make(splits["test"], False)

    model = MultiAspectClassifier(args.encoder, len(ASPECTS), len(CLASSES)).to(device)
    if not args.fine_tune:  # frozen-encoder baseline: train heads only
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder FROZEN (baseline: only the 6 heads train)")

    weights = class_weights(splits["train"])
    losses = [
        nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))
        for w in weights
    ]
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_macro, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (ids, mask, y) in enumerate(train_dl):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(ids, mask)
            loss = sum(losses[a](logits[a], y[:, a]) for a in range(len(ASPECTS)))
            loss.backward()
            opt.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step:4d}/{len(train_dl)}  "
                      f"loss={loss.item():.3f}")
        val_macro, _ = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: train_loss={running/len(train_dl):.3f}  "
              f"val_macroF1={val_macro:.3f}")
        if val_macro > best_macro:
            best_macro = val_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nBest val macro-F1: {best_macro:.3f}")

    test_macro, det = evaluate(model, test_dl, device)
    print(f"TEST macro-F1 (mean over aspects): {test_macro:.3f}\n")
    for aspect in ASPECTS:
        print(f"  {aspect:11} macroF1={det['per_aspect_f1'][aspect]:.3f}")
    print("\nPer-class report, Service (example aspect):")
    a = ASPECTS.index("Service")
    print(classification_report(
        det["gold"][a], det["preds"][a],
        labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    tok.save_pretrained(MODEL_DIR)
    (MODEL_DIR / "label_config.json").write_text(
        __import__("json").dumps(
            {"aspects": ASPECTS, "classes": CLASSES, "encoder": args.encoder},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved model -> {MODEL_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage-1 English ABSA model.")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--sources", nargs="+", default=["tripadvisor"],
                    choices=["tripadvisor", "booking"])
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap train docs (0 = all); use e.g. 3000 for a quick CPU run")
    ap.add_argument("--no-fine-tune", dest="fine_tune", action="store_false",
                    help="freeze the encoder, train only the heads (fast baseline)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
