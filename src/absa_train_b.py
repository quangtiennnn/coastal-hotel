"""
absa_train_b.py
===============
Step 4 of ABSA_TRAINING_PROPOSAL.md - train Model B and every ablation arm.

Model B trains on the **LLM silver labels over our own corpus** (GMap + Agoda,
bilingual), not on the human-annotated sets. That is the whole point of having
labeled 15k in-domain reviews: the human annotations are a *separate platform*
and are reserved for validation, where they double as a cross-platform
generalization test (see src/absa_eval.py).

    input   SENTENCE_LABELS  (59k rows / 42k sentences, vi + en)
    unit    sentence  - "train sentence-level, evaluate review-level"
            (open decision #1; sentence gives more examples and finer signal,
            and src/absa_eval.py aggregates back up to documents)
    model   xlm-roberta-base + 5 x 4-class heads (see src/absa_model.py)

The ablation arms are **pre-specified and always run** - none is gated on a
qualitative judgement like "if recall looks weak", which would make the choice
post-hoc and unfalsifiable:

    1 baseline   silver only, class weights ON          reference model
    2 augmented  silver + rare-cell top-up, weights ON  does targeted data help?
    3 no_weights baseline, class weights OFF            does weighting help?
    4 frozen     frozen encoder + linear heads          is fine-tuning worth it?

(Row 5 of the proposal's table, aggregation Rule B, is a re-scoring of arm 1 -
it needs no training run and lives in src/absa_eval.py.)

Usage:
    uv run python src/absa_train_b.py --arm baseline
    uv run python src/absa_train_b.py --arm all                # all 4 arms
    uv run python src/absa_train_b.py --arm baseline --max-train 4000 --epochs 1
    uv run python src/absa_train_b.py --stats                  # data only

NOTE on hardware (open decision #3): torch here is CPU-only. A full
xlm-roberta-base fine-tune over ~34k training sentences takes hours per arm on
CPU and minutes on one GPU - and the ablation table needs 4 runs. Use
--max-train for a local smoke run; do the reported runs on a GPU.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

import duckdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from absa_label import DB_PATH
from absa_model import (
    DEFAULT_ENCODER,
    MultiAspectClassifier,
    SentenceDataset,
    class_weights,
    make_collate,
    save_model,
)
from absa_validate import ASPECTS5, CLASS2ID, CLASSES, ID2CLASS

SUBSET_SEED = 20260727   # only used by the --max-train/--max-eval smoke caps

ARMS = {
    # arm name  -> (use train_aug, class weights, fine-tune encoder)
    "baseline":  (False, True,  True),
    "augmented": (True,  True,  True),
    "no_weights": (False, False, True),
    "frozen":    (False, True,  False),
    # The fourth cell of the weights x augmentation grid. Arms 2 and 3 each help
    # on human gold independently (+0.033 and +0.050 over arm 1), so the most
    # promising configuration was the only one never trained.
    "aug_no_weights": (True, False, True),
}


# ---------------------------------------------------------------------------
# SENTENCE_LABELS -> training rows
# ---------------------------------------------------------------------------

def load_sentence_rows(con: duckdb.DuckDBPyConnection, splits: list[str]
                       ) -> list[tuple[str, list[int], str]]:
    """[(sentence, [class_id per aspect], language), ...] for the given splits.

    One SENTENCE_LABELS row is one (sentence, overlapping aspect span) pair, so
    a sentence appears once per span that touches it. Collapsing them:

    - a sentence's label for an aspect is the MAJORITY sentiment over that
      aspect's spans on that sentence, ties -> neutral. This is deliberately
      the same convention as the silver review roll-up in absa_label.rollup(),
      so sentence and review labels never disagree by construction.
    - key_aspect 'none' rows are the programmatic explicit negatives from the
      bridge: the sentence stays, all five aspects are not_mentioned.
    - key_aspect 'other' rows (the dropped silver catch-all, 162 rows / 0.3%)
      contribute no label - it is not a real aspect. The sentence is kept, so
      the model still sees it as a negative example for the 5 real aspects.
    """
    placeholders = ", ".join(["?"] * len(splits))
    rows = con.execute(f"""
        SELECT review_id, sent_idx, sentence, key_aspect, sentiment, language
        FROM SENTENCE_LABELS
        WHERE split IN ({placeholders})
        ORDER BY review_id, sent_idx
    """, splits).fetchall()

    by_sent: dict[tuple[str, int], dict] = {}
    for review_id, sent_idx, sentence, key_aspect, sentiment, language in rows:
        key = (review_id, sent_idx)
        rec = by_sent.setdefault(key, {"text": sentence, "lang": language or "en",
                                       "votes": {a: [] for a in ASPECTS5}})
        if key_aspect in rec["votes"] and sentiment:
            rec["votes"][key_aspect].append(sentiment)

    out: list[tuple[str, list[int], str]] = []
    for rec in by_sent.values():
        labels = []
        for aspect in ASPECTS5:
            sents = rec["votes"][aspect]
            if not sents:
                labels.append(CLASS2ID["not_mentioned"])
                continue
            counts = Counter(sents)
            best = max(counts.values())
            winners = [s for s, n in counts.items() if n == best]
            labels.append(CLASS2ID[winners[0] if len(winners) == 1 else "neutral"])
        out.append((rec["text"], labels, rec["lang"]))
    return out


def describe(name: str, rows: list) -> None:
    langs = Counter(r[2] for r in rows)
    print(f"  {name:10} {len(rows):7,} sentences  "
          f"({', '.join(f'{l}={n:,}' for l, n in sorted(langs.items()))})")


def print_label_stats(rows: list, title: str) -> None:
    print(f"\n{title} - class distribution per aspect:")
    for a, aspect in enumerate(ASPECTS5):
        counts = Counter(r[1][a] for r in rows)
        line = "  ".join(f"{ID2CLASS[c][:3]}={counts.get(c, 0):6,}"
                         for c in range(len(CLASSES)))
        print(f"  {aspect:12} {line}")


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict]:
    """Macro-F1 per aspect (mean over the 4 classes), and their mean."""
    from sklearn.metrics import f1_score

    model.eval()
    preds = {a: [] for a in range(len(ASPECTS5))}
    gold = {a: [] for a in range(len(ASPECTS5))}
    for ids, mask, y in loader:
        logits = model(ids.to(device), mask.to(device))
        for a in range(len(ASPECTS5)):
            preds[a] += logits[a].argmax(1).cpu().tolist()
            gold[a] += y[:, a].tolist()
    per_aspect = {
        aspect: f1_score(gold[a], preds[a], average="macro", zero_division=0)
        for a, aspect in enumerate(ASPECTS5)
    }
    return sum(per_aspect.values()) / len(per_aspect), {
        "per_aspect_f1": per_aspect, "preds": preds, "gold": gold,
    }


def load_all(con: duckdb.DuckDBPyConnection) -> dict:
    """Every split an arm might need, read in one go.

    Read up front and held in memory so the DB connection can be CLOSED before
    training starts: DuckDB allows one writer, and a multi-hour fine-tune
    holding the file open locks out every notebook and every other script in
    the pipeline for its whole duration.
    """
    return {
        "train": load_sentence_rows(con, ["train"]),
        "train+aug": load_sentence_rows(con, ["train", "train_aug"]),
        "val": load_sentence_rows(con, ["val"]),
        "test": load_sentence_rows(con, ["test"]),
        "n_aug": con.execute(
            "SELECT count(*) FROM SENTENCE_LABELS WHERE split = 'train_aug'"
        ).fetchone()[0],
    }


def train_arm(data: dict, arm: str, args) -> dict:
    use_aug, weighted, fine_tune = ARMS[arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = data["train+aug"] if use_aug else data["train"]
    val_rows, test_rows = data["val"], data["test"]

    print(f"\n{'=' * 70}\nARM: {arm}   "
          f"(aug={use_aug}, class_weights={weighted}, fine_tune={fine_tune})\n"
          f"{'=' * 70}")
    print(f"Device: {device}  |  encoder: {args.encoder}")
    describe("train", train_rows)
    describe("val", val_rows)
    describe("test", test_rows)

    if use_aug and not data["n_aug"]:
        print("\n  !! The augmented arm found NO split='train_aug' sentences.\n"
              "     Run src/absa_augment.py, then LLM-label the new rows\n"
              "     (absa_label_submit/retrieve) and re-run src/absa_bridge.py.\n"
              "     Training now would just duplicate the baseline - skipped.")
        return {"arm": arm, "skipped": "no train_aug data"}

    # Caps are a fixed-seed random SAMPLE, not a head slice: rows come out
    # ordered by review_id, so the first N sentences would all come from the
    # same handful of reviews (and skew by language and hotel).
    if args.max_train and args.max_train < len(train_rows):
        train_rows = random.Random(SUBSET_SEED).sample(train_rows, args.max_train)
        print(f"  --max-train: sampled {len(train_rows):,} sentences "
              f"(SMOKE RUN - not reportable)")
    if args.max_eval:
        rng = random.Random(SUBSET_SEED)
        if args.max_eval < len(val_rows):
            val_rows = rng.sample(val_rows, args.max_eval)
        if args.max_eval < len(test_rows):
            test_rows = rng.sample(test_rows, args.max_eval)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder)
    collate = make_collate(tok, args.max_len)
    mk = lambda rows, shuffle: DataLoader(
        SentenceDataset(rows), batch_size=args.batch, shuffle=shuffle,
        collate_fn=collate,
    )
    train_dl, val_dl, test_dl = mk(train_rows, True), mk(val_rows, False), \
        mk(test_rows, False)

    model = MultiAspectClassifier(args.encoder).to(device)
    if not fine_tune:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  Encoder FROZEN - only the 5 linear heads train")

    if weighted:
        weights = class_weights(train_rows)
        losses = [nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float, device=device))
            for w in weights]
    else:
        losses = [nn.CrossEntropyLoss() for _ in ASPECTS5]
        print("  Class weights OFF (ablation row 3)")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_macro, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for step, (ids, mask, y) in enumerate(train_dl):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(ids, mask)
            loss = sum(losses[a](logits[a], y[:, a]) for a in range(len(ASPECTS5)))
            loss.backward()
            opt.step()
            running += loss.item()
            if step % args.log_every == 0:
                rate = (step + 1) * args.batch / max(time.time() - t0, 1e-9)
                print(f"  epoch {epoch} step {step:5d}/{len(train_dl)}  "
                      f"loss={loss.item():.3f}  ({rate:.0f} sent/s)", flush=True)
        val_macro, _ = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl),
                        "val_macro_f1": val_macro})
        print(f"  epoch {epoch}: train_loss={running / len(train_dl):.3f}  "
              f"val_macroF1={val_macro:.3f}  ({time.time() - t0:.0f}s)")
        if val_macro > best_macro:
            best_macro = val_macro
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Silver held-out performance. NOT the paper's headline number - that is the
    # human-gold score in src/absa_eval.py. This only says how well the student
    # reproduces its own teacher on unseen in-domain sentences.
    test_macro, det = evaluate(model, test_dl, device)
    print(f"\n  silver held-out macro-F1 (mean over aspects): {test_macro:.3f}")
    for aspect in ASPECTS5:
        print(f"    {aspect:12} {det['per_aspect_f1'][aspect]:.3f}")

    meta = {
        "arm": arm, "encoder": args.encoder, "use_train_aug": use_aug,
        "class_weights": weighted, "fine_tune": fine_tune,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "max_len": args.max_len, "n_train": len(train_rows),
        "n_val": len(val_rows), "n_test": len(test_rows),
        "max_train_cap": args.max_train or None,
        "best_val_macro_f1": best_macro,
        "silver_test_macro_f1": test_macro,
        "silver_test_per_aspect_f1": det["per_aspect_f1"],
        "history": history,
    }
    out = save_model(model, tok, arm, meta)
    print(f"  saved -> {out}")
    return meta


def main() -> None:
    # line_buffering: a training run is long, and its progress is worthless if
    # it only appears when the process exits (Python block-buffers a pipe).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                           line_buffering=True)
    ap = argparse.ArgumentParser(description="Train Model B + ablation arms.")
    ap.add_argument("--arm", default="baseline",
                    choices=[*ARMS, "all"], help="ablation arm to train")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap training sentences (0 = all) - for CPU smoke runs")
    ap.add_argument("--max-eval", type=int, default=0,
                    help="cap val/test sentences (0 = all)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--stats", action="store_true",
                    help="print the training data only, train nothing")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    data = load_all(con)
    con.close()   # released before training - see load_all()

    if args.stats:
        for name in ("train", "train+aug", "val", "test"):
            describe(name, data[name])
        print_label_stats(data["train"], "TRAIN (silver)")
        return

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    results = [train_arm(data, arm, args) for arm in arms]

    print(f"\n{'=' * 70}\nSummary (silver held-out - see src/absa_eval.py for the "
          f"human-gold table)\n{'=' * 70}")
    for r in results:
        if "skipped" in r:
            print(f"  {r['arm']:12} SKIPPED ({r['skipped']})")
        else:
            print(f"  {r['arm']:12} val={r['best_val_macro_f1']:.3f}  "
                  f"silver_test={r['silver_test_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
