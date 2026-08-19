"""
absa_train_cascade.py
=====================
EXPERIMENTAL - two-stage cascade variant of Model B. NOT part of the
pre-specified ablation table in ABSA_TRAINING_PROPOSAL.md.

    stage 1  detection  5 x Linear(768 -> 2)   not_mentioned / present
    stage 2  polarity   5 x Linear(768 -> P)   negative / [neutral] / positive
                                               loss MASKED to present cells only

The point of the split (REVIEW_ABSA_MODEL_B.md 2.3, ARCHITECTURE_ABSA_MODEL_B.md
2): in the joint 4-class model the polarity decision shares a softmax with a
class that is 68.5-92.8% of all training cells. Masking the polarity loss to
present cells is the only way to find out whether that flood is actually costing
anything - and the detection stage gets a tunable threshold for free, which the
joint argmax cannot have.

-----------------------------------------------------------------------------
DELETE-SAFE. This experiment writes NOTHING into hotel_reviews.db and touches no
existing model directory. To remove every trace of it:

    rm src/absa_train_cascade.py
    rm -rf models/absa_cascade/
    rm data/absa_results/cascade_*.json

The DB is opened READ-ONLY here. ABSA_SENT_PRED / ABSA_EVAL_RESULTS are never
written, so nothing in RESULTS_ABSA_MODEL_B.md can be perturbed by this file.
-----------------------------------------------------------------------------

Usage:
    # smoke run first - verifies the whole path in ~5 min on CPU
    uv run python src/absa_train_cascade.py --name smoke --max-train 2000 --epochs 1

    # real run (~2h45 CPU, minutes on a GPU)
    uv run python src/absa_train_cascade.py --name cascade3

    # drop neutral entirely (the 3-class decision of REVIEW 3.1)
    uv run python src/absa_train_cascade.py --name cascade2 --polarity-classes 2

    # score against the human gold dev partition (~30 min CPU) and sweep tau
    uv run python src/absa_train_cascade.py --name cascade3 --dev --tune-threshold

Comparison targets (arm 3 `no_weights`, the best joint arm):
    silver held-out 4-class macro-F1   0.746
    human gold dev  4-class macro-F1   0.569
    human gold dev  presence-F1        0.873
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from absa_bridge import segment_review
from absa_label import DB_PATH
from absa_model import DEFAULT_ENCODER, SentenceDataset, aggregate_review, make_collate
from absa_train_b import load_sentence_rows
from absa_validate import ASPECTS5, CLASSES, ID2CLASS

ROOT = Path(__file__).parent.parent
MODEL_ROOT = ROOT / "models" / "absa_cascade"      # separate from models/absa_b
RESULTS_DIR = ROOT / "data" / "absa_results"
SUBSET_SEED = 20260727       # same seed as absa_train_b, for comparable caps

# Composition back to the 4-class label space, so every number this file prints
# is directly comparable to the joint model's. CLASSES = [nm, neg, neu, pos].
POL_TO_C4 = {
    3: [1, 2, 3],    # negative, neutral, positive
    2: [1, 3],       # negative, positive          (neutral dropped from training)
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CascadeClassifier(nn.Module):
    """Shared encoder, then TWO head groups per aspect instead of one.

    Same encoder and same pooling as MultiAspectClassifier, so the only
    difference from the joint model is how the decision is factorised. That is
    deliberate: if the cascade wins, the win has to come from the factorisation
    and not from a bigger or different encoder.
    """

    def __init__(self, encoder_name: str = DEFAULT_ENCODER,
                 n_aspects: int = len(ASPECTS5), n_polarity: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.n_polarity = n_polarity
        self.det_heads = nn.ModuleList(
            [nn.Linear(hidden, 2) for _ in range(n_aspects)])
        self.pol_heads = nn.ModuleList(
            [nn.Linear(hidden, n_polarity) for _ in range(n_aspects)])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return ([h(cls) for h in self.det_heads],
                [h(cls) for h in self.pol_heads])


# ---------------------------------------------------------------------------
# 4-class silver labels -> (detection, polarity) targets
# ---------------------------------------------------------------------------

def split_targets(y: torch.Tensor, n_polarity: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 5) of 4-class ids -> detection (B, 5) and polarity (B, 5).

    Polarity uses ignore_index -100 for every cell the polarity stage must not
    see: not_mentioned always, and neutral too when --polarity-classes 2. That
    masking IS the experiment - it is what stops the 68-93% not_mentioned
    majority from reaching the polarity heads at all.
    """
    det = (y > 0).long()
    pol = torch.full_like(y, -100)
    pol[y == 1] = 0                       # negative
    if n_polarity == 3:
        pol[y == 2] = 1                   # neutral
        pol[y == 3] = 2                   # positive
    else:
        pol[y == 3] = 1                   # positive; neutral stays ignored
    return det, pol


def detection_weights(rows) -> list[list[float]]:
    """Inverse-frequency weights for the BINARY detection task, per aspect."""
    out = []
    for a in range(len(ASPECTS5)):
        counts = Counter(1 if r[1][a] > 0 else 0 for r in rows)
        total = sum(counts.values())
        out.append([total / (2 * counts[c]) if counts.get(c) else 0.0
                    for c in (0, 1)])
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, n_polarity: int, tau: float = 0.5) -> dict:
    """Detection / polarity / composed-4-class, all on the same pass.

    The composed 4-class number is the one that is comparable to the joint
    model. Detection and polarity are reported alongside it because they are
    what the cascade exists to separate - and because the joint model can only
    produce them by post-hoc decomposition.
    """
    from sklearn.metrics import f1_score

    model.eval()
    n_a = len(ASPECTS5)
    det_p = {a: [] for a in range(n_a)}
    det_g = {a: [] for a in range(n_a)}
    pol_p = {a: [] for a in range(n_a)}
    pol_g = {a: [] for a in range(n_a)}
    c4_p = {a: [] for a in range(n_a)}
    c4_g = {a: [] for a in range(n_a)}

    for ids, mask, y in loader:
        det_logits, pol_logits = model(ids.to(device), mask.to(device))
        det_t, pol_t = split_targets(y, n_polarity)
        for a in range(n_a):
            present = (det_logits[a].softmax(1)[:, 1] > tau).cpu()
            pol_hat = pol_logits[a].argmax(1).cpu()
            det_p[a] += present.long().tolist()
            det_g[a] += det_t[:, a].tolist()
            keep = pol_t[:, a] != -100
            if keep.any():
                pol_p[a] += pol_hat[keep].tolist()
                pol_g[a] += pol_t[keep, a].tolist()
            mapping = POL_TO_C4[n_polarity]
            c4_p[a] += [0 if not pr else mapping[ph]
                        for pr, ph in zip(present.tolist(), pol_hat.tolist())]
            c4_g[a] += y[:, a].tolist()

    det_f1, pol_f1, c4_f1 = {}, {}, {}
    for a, aspect in enumerate(ASPECTS5):
        det_f1[aspect] = f1_score(det_g[a], det_p[a], zero_division=0)
        # Explicit labels: without them sklearn averages only over the classes
        # that actually occur, so an aspect whose test split contains no neutral
        # cell (loyalty/neutral has 15 rows in all of training) would silently
        # be scored over 2 classes while the others are scored over 3 - and the
        # numbers would not be comparable across aspects or arms.
        pol_f1[aspect] = (f1_score(pol_g[a], pol_p[a],
                                   labels=list(range(model.n_polarity)),
                                   average="macro", zero_division=0)
                          if pol_g[a] else 0.0)
        c4_f1[aspect] = f1_score(c4_g[a], c4_p[a], labels=list(range(4)),
                                 average="macro", zero_division=0)
    mean = lambda d: sum(d.values()) / len(d)
    return {
        "tau": tau,
        "detection_f1": det_f1, "detection_mean": mean(det_f1),
        "polarity_macro_f1": pol_f1, "polarity_mean": mean(pol_f1),
        "composed_4class_f1": c4_f1, "composed_4class_mean": mean(c4_f1),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(data: dict, args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_pol = args.polarity_classes
    train_rows, val_rows, test_rows = data["train"], data["val"], data["test"]

    print(f"\n{'=' * 70}\nCASCADE: {args.name}   "
          f"(polarity_classes={n_pol}, det_weights={args.class_weights}, "
          f"lambda={args.pol_weight})\n{'=' * 70}")
    print(f"Device: {device}  |  encoder: {args.encoder}")

    if args.max_train and args.max_train < len(train_rows):
        train_rows = random.Random(SUBSET_SEED).sample(train_rows, args.max_train)
        print(f"  --max-train: sampled {len(train_rows):,} sentences "
              f"(SMOKE RUN - not reportable)")
    if args.max_eval:
        rng = random.Random(SUBSET_SEED)
        val_rows = rng.sample(val_rows, min(args.max_eval, len(val_rows)))
        test_rows = rng.sample(test_rows, min(args.max_eval, len(test_rows)))
    for nm, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        langs = Counter(r[2] for r in rows)
        print(f"  {nm:6} {len(rows):7,} sentences  "
              f"({', '.join(f'{l}={n:,}' for l, n in sorted(langs.items()))})")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    collate = make_collate(tok, args.max_len)
    mk = lambda rows, sh: DataLoader(SentenceDataset(rows), batch_size=args.batch,
                                     shuffle=sh, collate_fn=collate)
    train_dl, val_dl, test_dl = mk(train_rows, True), mk(val_rows, False), \
        mk(test_rows, False)

    model = CascadeClassifier(args.encoder, n_polarity=n_pol).to(device)

    if args.class_weights:
        dw = detection_weights(train_rows)
        det_losses = [nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float, device=device)) for w in dw]
    else:
        det_losses = [nn.CrossEntropyLoss() for _ in ASPECTS5]
        print("  Detection class weights OFF (matches arm 3, the best joint arm)")
    # ignore_index=-100 is what masks the polarity loss to present cells.
    pol_losses = [nn.CrossEntropyLoss(ignore_index=-100) for _ in ASPECTS5]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for step, (ids, mask, y) in enumerate(train_dl):
            ids, mask = ids.to(device), mask.to(device)
            det_t, pol_t = split_targets(y, n_pol)
            det_t, pol_t = det_t.to(device), pol_t.to(device)
            opt.zero_grad()
            det_logits, pol_logits = model(ids, mask)
            loss = 0.0
            for a in range(len(ASPECTS5)):
                loss = loss + det_losses[a](det_logits[a], det_t[:, a])
                # A batch can legitimately contain no present cell for a rare
                # aspect (loyalty is absent in 92.8% of sentences). CE with
                # every target ignored returns nan, which would poison the
                # whole step - so the term is simply skipped.
                if (pol_t[:, a] != -100).any():
                    loss = loss + args.pol_weight * pol_losses[a](
                        pol_logits[a], pol_t[:, a])
            loss.backward()
            opt.step()
            running += loss.detach().item()
            if step % args.log_every == 0:
                rate = (step + 1) * args.batch / max(time.time() - t0, 1e-9)
                print(f"  epoch {epoch} step {step:5d}/{len(train_dl)}  "
                      f"loss={loss.detach().item():.3f}  ({rate:.0f} sent/s)",
                      flush=True)

        ev = evaluate(model, val_dl, device, n_pol)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl),
                        "val_composed_4class": ev["composed_4class_mean"],
                        "val_detection": ev["detection_mean"],
                        "val_polarity": ev["polarity_mean"]})
        print(f"  epoch {epoch}: loss={running / len(train_dl):.3f}  "
              f"val 4class={ev['composed_4class_mean']:.3f}  "
              f"det={ev['detection_mean']:.3f}  pol={ev['polarity_mean']:.3f}  "
              f"({time.time() - t0:.0f}s)")
        # Selected on the COMPOSED number so the checkpoint rule matches the
        # joint model's (best silver val macro-F1) and the arms stay comparable.
        if ev["composed_4class_mean"] > best:
            best = ev["composed_4class_mean"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_ev = evaluate(model, test_dl, device, n_pol)
    print(f"\n  SILVER HELD-OUT (compare: joint arm 3 = 0.746)")
    print(f"    composed 4-class : {test_ev['composed_4class_mean']:.3f}")
    print(f"    detection  (bin) : {test_ev['detection_mean']:.3f}")
    print(f"    polarity (masked): {test_ev['polarity_mean']:.3f}")
    for aspect in ASPECTS5:
        print(f"      {aspect:12} 4cls={test_ev['composed_4class_f1'][aspect]:.3f}"
              f"  det={test_ev['detection_f1'][aspect]:.3f}"
              f"  pol={test_ev['polarity_macro_f1'][aspect]:.3f}")

    meta = {
        "name": args.name, "kind": "cascade", "encoder": args.encoder,
        "polarity_classes": n_pol, "detection_class_weights": args.class_weights,
        "pol_loss_weight": args.pol_weight, "epochs": args.epochs, "lr": args.lr,
        "batch": args.batch, "max_len": args.max_len,
        "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "max_train_cap": args.max_train or None,
        "best_val_composed_4class": best,
        "silver_test": test_ev, "history": history,
    }
    out = MODEL_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tok.save_pretrained(out)
    (out / "config.json").write_text(
        json.dumps({"aspects": ASPECTS5, "classes": CLASSES, **meta},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {out}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"cascade_{args.name}_silver.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def load_trained(name: str, device):
    path = MODEL_ROOT / name
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = CascadeClassifier(cfg["encoder"], n_polarity=cfg["polarity_classes"])
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(str(path)), cfg


# ---------------------------------------------------------------------------
# Human-gold dev scoring (read-only; nothing is written to the DB)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_dev(con, name: str, batch: int, max_len: int):
    """Segment VALIDATE-dev, predict, and return per-sentence probabilities.

    Probabilities - not argmax labels - because the whole reason for the
    detection stage is that tau becomes tunable. Kept in memory and never
    written to ABSA_SENT_PRED, so this experiment stays deletable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_trained(name, device)
    n_pol = cfg["polarity_classes"]

    docs = con.execute(
        "SELECT doc_id, review_text, language FROM ABSA_VALIDATE "
        "WHERE partition = 'dev'").fetchall()
    flat = []
    for doc_id, text, lang in docs:
        sents = segment_review(text, lang or "en")
        if not sents:
            sents = [(0, len(text), text.strip())]
        for idx, (_, _, s) in enumerate(sents):
            flat.append((doc_id, idx, s))
    print(f"  {len(docs):,} dev docs -> {len(flat):,} sentences "
          f"({len(flat) / len(docs):.2f} per doc)")

    p_present = {a: [] for a in ASPECTS5}
    pol_hat = {a: [] for a in ASPECTS5}
    t0 = time.time()
    for i in range(0, len(flat), batch):
        chunk = [c[2] for c in flat[i:i + batch]]
        enc = tok(chunk, truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        det_logits, pol_logits = model(enc["input_ids"], enc["attention_mask"])
        for a, aspect in enumerate(ASPECTS5):
            p_present[aspect] += det_logits[a].softmax(1)[:, 1].cpu().tolist()
            pol_hat[aspect] += pol_logits[a].argmax(1).cpu().tolist()
        if i % (batch * 50) == 0:
            done = i + len(chunk)
            print(f"    {done:,}/{len(flat):,}  "
                  f"({done / max(time.time() - t0, 1e-9):.0f} sent/s)", flush=True)
    return [(d, s) for d, s, _ in flat], p_present, pol_hat, n_pol


def score_dev(con, keys, p_present, pol_hat, n_pol, taus: dict) -> dict:
    """Threshold -> compose -> aggregate_review -> 4-class macro-F1 vs gold."""
    from sklearn.metrics import f1_score

    gold = {r[0]: dict(zip(ASPECTS5, r[1:])) for r in con.execute(
        "SELECT doc_id, " + ", ".join(f"asp5_{a}" for a in ASPECTS5) +
        " FROM ABSA_VALIDATE WHERE partition = 'dev'").fetchall()}
    mapping = POL_TO_C4[n_pol]

    per_doc: dict[str, dict[str, list[str]]] = {}
    for i, (doc_id, _) in enumerate(keys):
        d = per_doc.setdefault(doc_id, {a: [] for a in ASPECTS5})
        for aspect in ASPECTS5:
            present = p_present[aspect][i] > taus[aspect]
            d[aspect].append(ID2CLASS[mapping[pol_hat[aspect][i]]] if present
                             else "not_mentioned")

    out, pres = {}, {}
    for aspect in ASPECTS5:
        g = [gold[d][aspect] for d in per_doc]
        p = [aggregate_review(per_doc[d], aspect, "primary") for d in per_doc]
        out[aspect] = f1_score(g, p, labels=CLASSES, average="macro",
                               zero_division=0)
        pres[aspect] = f1_score([x != "not_mentioned" for x in g],
                                [x != "not_mentioned" for x in p], zero_division=0)
    return {"per_aspect_4class": out, "mean_4class": sum(out.values()) / len(out),
            "per_aspect_presence": pres,
            "mean_presence": sum(pres.values()) / len(pres),
            "taus": dict(taus), "n_docs": len(per_doc)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description="Cascade variant (EXPERIMENTAL).")
    ap.add_argument("--name", default="cascade3", help="run name -> models/absa_cascade/<name>")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--polarity-classes", type=int, default=3, choices=[2, 3],
                    help="3 = neg/neu/pos; 2 = neg/pos, neutral dropped from training")
    ap.add_argument("--class-weights", action="store_true",
                    help="inverse-frequency weights on the DETECTION heads")
    ap.add_argument("--pol-weight", type=float, default=1.0,
                    help="lambda on the polarity loss term")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dev", action="store_true",
                    help="score against human gold VALIDATE-dev (no training)")
    ap.add_argument("--tune-threshold", action="store_true",
                    help="sweep tau per aspect on dev (requires --dev)")
    args = ap.parse_args()

    if args.dev:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        keys, p_present, pol_hat, n_pol = predict_dev(
            con, args.name, args.batch, args.max_len)
        base = score_dev(con, keys, p_present, pol_hat, n_pol,
                         {a: 0.5 for a in ASPECTS5})
        print(f"\n  HUMAN GOLD DEV, tau=0.5 (compare: joint arm 3 = 0.569 / 0.873)")
        print(f"    4-class  : {base['mean_4class']:.3f}")
        print(f"    presence : {base['mean_presence']:.3f}")
        for a in ASPECTS5:
            print(f"      {a:12} 4cls={base['per_aspect_4class'][a]:.3f}  "
                  f"pres={base['per_aspect_presence'][a]:.3f}")

        result = {"name": args.name, "tau_0.5": base}
        if args.tune_threshold:
            # Legitimate dev work - the proposal pre-specifies "choosing
            # thresholds" as a VALIDATE-dev activity. The test partition is
            # never read here.
            print("\n  sweeping tau per aspect on dev ...")
            best_taus = {}
            for aspect in ASPECTS5:
                rows = []
                for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    taus = {**{a: 0.5 for a in ASPECTS5}, aspect: t}
                    s = score_dev(con, keys, p_present, pol_hat, n_pol, taus)
                    rows.append((t, s["per_aspect_4class"][aspect]))
                bt, bf = max(rows, key=lambda r: r[1])
                best_taus[aspect] = bt
                print(f"    {aspect:12} " +
                      "  ".join(f"{t}:{f:.3f}" for t, f in rows) +
                      f"   -> tau={bt} ({bf:.3f})")
            tuned = score_dev(con, keys, p_present, pol_hat, n_pol, best_taus)
            print(f"\n  TUNED taus {best_taus}")
            print(f"    4-class  : {tuned['mean_4class']:.3f}")
            print(f"    presence : {tuned['mean_presence']:.3f}")
            result["tuned"] = tuned
        con.close()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"cascade_{args.name}_dev.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)
    data = {"train": load_sentence_rows(con, ["train"]),
            "val": load_sentence_rows(con, ["val"]),
            "test": load_sentence_rows(con, ["test"])}
    con.close()      # released before the multi-hour run, as in absa_train_b
    train(data, args)


if __name__ == "__main__":
    main()
