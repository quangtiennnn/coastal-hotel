"""
absa_train_sub.py
=================
Stage 2 - the sub-aspect head. Implements
[markdown/PROPOSAL_SUB_ASPECT_HEAD.md](../markdown/PROPOSAL_SUB_ASPECT_HEAD.md).

    encoder da fine-tune (arm no_weights / aug_no_weights)   <- FROZEN
      -> Linear(768 -> 34)
      -> mask theo SUB_TO_KEY: chi tinh logit thuoc key_aspect dang hoi
      -> sigmoid -> masked BCE          (multi-label: 12.9% cell co >=2 nhan)

Three corrections to the v3 spec in ABSA_TRAINING_PROPOSAL.md, all argued in the
proposal above:

  1. class weighting defaults OFF (v3 said on). Inverse-frequency weighting was
     measured to HURT twice - the 2x2 main-head grid and the cascade detection
     head - the second of which isolates polarity completely, so the harm is not
     about one weight vector serving two tasks.
  2. the 0.55 stop-gradient fallback threshold is DROPPED. It was a magic number
     with no gold, no precedent and no baseline to anchor it. A majority-class
     floor is reported instead.
  3. metrics are macro-F1 / precision@1 / mAP, not "top-1 / top-2 accuracy" -
     top-k accuracy is undefined against multi-label ground truth.

Because the encoder is frozen, every sentence is encoded ONCE and the 768-dim
CLS vectors are cached; the head is a single Linear, so training it is seconds
rather than minutes. The same cached vectors also drive the main heads, which is
how the predicted-conditioned evaluation is computed without a second pass.

-----------------------------------------------------------------------------
DELETE-SAFE. Writes NOTHING into hotel_reviews.db (opened READ-ONLY) and touches
no existing model directory. To remove every trace:

    rm src/absa_train_sub.py
    rm -rf models/absa_sub/
    rm -rf data/sub_check/
    rm data/absa_results/sub_*.json

Main-head predictions stay bit-identical: the encoder is frozen and the 5 main
heads are never updated, so every figure in RESULTS_ABSA_MODEL_B_V2.md stands
and the VALIDATE-test freeze is untouched.
-----------------------------------------------------------------------------

Usage:
    # train the head on both reported encoders (~5 min each on GPU)
    uv run python src/absa_train_sub.py --arm no_weights
    uv run python src/absa_train_sub.py --arm aug_no_weights

    # the 2x2 ablation the proposal asks for
    uv run python src/absa_train_sub.py --arm no_weights --class-weights
    uv run python src/absa_train_sub.py --arm no_weights --use-aug

    # draw the 200-pair human check, coastal_access oversampled (no model needed)
    uv run python src/absa_train_sub.py --sample-check --coastal-n 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

from absa_label import DB_PATH, SUB_ASPECT_CHOICES, SUB_TO_KEY

# torch / transformers / absa_model are imported INSIDE train(), not here: the
# --sample-check path needs neither, and it is the one item on the critical path
# that depends on human availability. Keeping it torch-free means it still runs
# when the local torch install is unavailable (Windows Smart App Control blocks
# the unsigned CUDA DLLs) - see PROPOSAL_SUB_ASPECT_HEAD.md.
ASPECTS5 = ["facility", "amenity", "service", "experience", "loyalty"]

ROOT = Path(__file__).parent.parent
MODEL_ROOT = ROOT / "models" / "absa_sub"
RESULTS_DIR = ROOT / "data" / "absa_results"
CHECK_DIR = ROOT / "data" / "sub_check"
SEED = 20260805

# Fixed order for the 34 outputs, grouped by macro aspect so the mask is a slice.
SUB_LIST: list[str] = [s for a in ASPECTS5 for s in SUB_ASPECT_CHOICES[a]]
SUB2IDX = {s: i for i, s in enumerate(SUB_LIST)}
ASPECT_SUBS = {a: [SUB2IDX[s] for s in SUB_ASPECT_CHOICES[a]] for a in ASPECTS5}


def aspect_masks(device):
    """aspect -> bool(34). Enforces the hierarchy structurally, not by learning:
    a `service` sentence can never be assigned `coastal_view`."""
    import torch
    out = {}
    for a in ASPECTS5:
        m = torch.zeros(len(SUB_LIST), dtype=torch.bool, device=device)
        m[ASPECT_SUBS[a]] = True
        out[a] = m
    return out


# ---------------------------------------------------------------------------
# Data: one training example = one (sentence, key_aspect) CELL, multi-hot target
# ---------------------------------------------------------------------------

def load_cells(con, splits: list[str]) -> list[dict]:
    """[{sentence, aspect, subs:set[str]}, ...] for the given splits.

    Only cells that pass the proposal's loss gate are returned: the main label
    is not not_mentioned (guaranteed here, since a row exists only where the
    labeler found the aspect) and a silver sub_aspect exists. `none` and `other`
    rows contribute nothing, so the aux task is never flooded by the same
    not_mentioned majority that dominates the main task.
    """
    ph = ", ".join(["?"] * len(splits))
    rows = con.execute(f"""
        SELECT review_id, sent_idx, sentence, key_aspect, sub_aspect
        FROM SENTENCE_LABELS
        WHERE split IN ({ph})
          AND sub_aspect IS NOT NULL
          AND key_aspect NOT IN ('none', 'other')
        ORDER BY review_id, sent_idx
    """, splits).fetchall()

    cells: dict[tuple, dict] = {}
    for review_id, sent_idx, sentence, aspect, sub in rows:
        if sub not in SUB2IDX:          # coined sub-aspects under `other`
            continue
        key = (review_id, sent_idx, aspect)
        c = cells.setdefault(key, {"sentence": sentence, "aspect": aspect,
                                   "subs": set()})
        c["subs"].add(sub)
    return [c for c in cells.values() if c["subs"]]


def encode(model, tok, sentences: list[str], device, batch: int = 128,
           max_len: int = 128) -> torch.Tensor:
    """Distinct sentences -> cached CLS vectors. One pass, encoder frozen."""
    import torch
    model.eval()
    out = []
    t0 = time.time()
    for i in range(0, len(sentences), batch):
        chunk = sentences[i:i + batch]
        enc = tok(chunk, truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        h = model.encoder(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"])
        out.append(h.last_hidden_state[:, 0].cpu())
        if i % (batch * 40) == 0:
            done = i + len(chunk)
            print(f"    encode {done:,}/{len(sentences):,} "
                  f"({done / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)
    return torch.cat(out)


def build_tensors(cells: list[dict], cls_of: dict[str, int], cls_bank):
    """cells -> (X, Y multi-hot, aspect_idx)."""
    import torch
    idx = torch.tensor([cls_of[c["sentence"]] for c in cells])
    X = cls_bank[idx]
    Y = torch.zeros(len(cells), len(SUB_LIST))
    for i, c in enumerate(cells):
        for s in c["subs"]:
            Y[i, SUB2IDX[s]] = 1.0
    A = torch.tensor([ASPECTS5.index(c["aspect"]) for c in cells])
    return X, Y, A


# ---------------------------------------------------------------------------
# Metrics - multi-label, per the proposal's correction 3
# ---------------------------------------------------------------------------

def evaluate(logits, Y, A, masks, majority, covered=None) -> dict:
    """macro-F1 / precision@1 / mAP per macro aspect, plus a majority floor.

    `covered` (optional) is a bool per cell saying whether the MAIN heads also
    predicted this aspect present. When given, cells the main heads missed are
    scored as misses - that is the honest end-to-end number, as opposed to the
    gold-conditioned one which isolates the sub-aspect task itself.
    """
    import torch
    from sklearn.metrics import f1_score, average_precision_score

    res = {}
    for ai, aspect in enumerate(ASPECTS5):
        sel = (A == ai)
        n = int(sel.sum())
        if n == 0:
            continue
        cols = ASPECT_SUBS[aspect]
        lg, y = logits[sel][:, cols], Y[sel][:, cols]
        ok = covered[sel] if covered is not None else torch.ones(n, dtype=torch.bool)

        # precision@1: the single most confident sub is in the gold set
        top1 = lg.argmax(1)
        hit = y[torch.arange(n), top1] > 0
        p_at_1 = float((hit & ok).float().mean())

        # macro-F1 over this aspect's subs, threshold 0.5 on the sigmoid
        pred = (torch.sigmoid(lg) > 0.5).int()
        if covered is not None:
            pred[~ok] = 0
        f1 = f1_score(y.int().numpy(), pred.numpy(), average="macro",
                      zero_division=0)

        # mAP: ranking quality, respects multi-label gold
        aps = []
        for i in range(n):
            if y[i].sum() == 0:
                continue
            s = lg[i].numpy().copy()
            if covered is not None and not bool(ok[i]):
                s = -s          # a missed aspect ranks its subs no better than chance
            aps.append(average_precision_score(y[i].int().numpy(), s))
        mAP = float(sum(aps) / len(aps)) if aps else 0.0

        # floor: always emit this aspect's most frequent sub in training
        mj = majority[aspect]
        base = float((y[:, cols.index(mj)] > 0).float().mean())

        res[aspect] = {"n_cells": n, "precision@1": p_at_1, "macro_f1": f1,
                       "mAP": mAP, "baseline_p@1": base,
                       "coverage": float(ok.float().mean())}
    mean = lambda k: sum(v[k] for v in res.values()) / len(res)
    return {"per_aspect": res, "mean_precision@1": mean("precision@1"),
            "mean_macro_f1": mean("macro_f1"), "mean_mAP": mean("mAP"),
            "mean_baseline_p@1": mean("baseline_p@1")}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args) -> dict:
    import torch
    import torch.nn as nn
    from absa_model import load_model
    from absa_validate import ID2CLASS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    splits = ["train", "train_aug"] if args.use_aug else ["train"]
    tr = load_cells(con, splits)
    va = load_cells(con, ["val"])
    te = load_cells(con, ["test"])
    con.close()

    print(f"\n{'=' * 70}\nSUB-ASPECT HEAD on encoder '{args.arm}'"
          f"   (aug={args.use_aug}, class_weights={args.class_weights})\n{'=' * 70}")
    print(f"Device: {device}")
    for nm, c in (("train", tr), ("val", va), ("test", te)):
        multi = sum(1 for x in c if len(x["subs"]) > 1)
        print(f"  {nm:6} {len(c):7,} cells  ({multi / len(c) * 100:.1f}% multi-label)")

    model, tok, cfg = load_model(args.arm, device)
    for p in model.parameters():
        p.requires_grad = False          # stop-gradient: nothing reported changes

    sents = sorted({c["sentence"] for c in tr + va + te})
    print(f"  encoding {len(sents):,} distinct sentences (frozen encoder)")
    bank = encode(model, tok, sents, device, args.batch, args.max_len)
    cls_of = {s: i for i, s in enumerate(sents)}

    Xtr, Ytr, Atr = build_tensors(tr, cls_of, bank)
    Xva, Yva, Ava = build_tensors(va, cls_of, bank)
    Xte, Yte, Ate = build_tensors(te, cls_of, bank)

    masks = aspect_masks("cpu")
    counts = Counter()
    for c in tr:
        counts.update(c["subs"])
    majority = {a: max(ASPECT_SUBS[a],
                       key=lambda i: counts.get(SUB_LIST[i], 0)) for a in ASPECTS5}

    head = nn.Linear(bank.shape[1], len(SUB_LIST)).to(device)
    if args.class_weights:
        # within-aspect inverse frequency, as pos_weight on the BCE
        pw = torch.ones(len(SUB_LIST))
        for a in ASPECTS5:
            tot = sum(counts.get(SUB_LIST[i], 0) for i in ASPECT_SUBS[a]) or 1
            for i in ASPECT_SUBS[a]:
                pw[i] = tot / (len(ASPECT_SUBS[a]) * max(counts.get(SUB_LIST[i], 0), 1))
        crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device), reduction="none")
        print("  class weights ON (ablation - default is OFF, see proposal 6.2)")
    else:
        crit = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    mask_mat = torch.stack([masks[a] for a in ASPECTS5]).float().to(device)
    Xtr_d, Ytr_d, Atr_d = Xtr.to(device), Ytr.to(device), Atr.to(device)

    best, best_state, history = -1.0, None, []
    n = len(Xtr_d)
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, args.head_batch):
            b = perm[i:i + args.head_batch]
            m = mask_mat[Atr_d[b]]                  # (B, 34) 1 where scorable
            loss = (crit(head(Xtr_d[b]), Ytr_d[b]) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.detach())
        head.eval()
        with torch.no_grad():
            lv = head(Xva.to(device)).cpu()
        ev = evaluate(lv, Yva, Ava, masks, majority)
        history.append({"epoch": epoch, "loss": running / max(n // args.head_batch, 1),
                        "val_p@1": ev["mean_precision@1"],
                        "val_macro_f1": ev["mean_macro_f1"]})
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  loss={history[-1]['loss']:.4f}  "
                  f"val P@1={ev['mean_precision@1']:.3f}  "
                  f"macroF1={ev['mean_macro_f1']:.3f}", flush=True)
        if ev["mean_precision@1"] > best:
            best = ev["mean_precision@1"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state:
        head.load_state_dict(best_state)

    # --- silver held-out, two conditionings -------------------------------
    head.eval()
    with torch.no_grad():
        lt = head(Xte.to(device)).cpu()
        main = [h(Xte.to(device)).argmax(1).cpu() for h in model.heads]
    gold_cond = evaluate(lt, Yte, Ate, masks, majority)
    covered = torch.tensor([
        ID2CLASS[int(main[int(Ate[i])][i])] != "not_mentioned" for i in range(len(Ate))])
    pred_cond = evaluate(lt, Yte, Ate, masks, majority, covered=covered)

    print(f"\n  SILVER HELD-OUT  ({len(te):,} cells)")
    print(f"  {'':12}{'P@1':>8}{'macroF1':>9}{'mAP':>7}{'floor':>8}")
    print(f"  {'gold-cond':12}{gold_cond['mean_precision@1']:>8.3f}"
          f"{gold_cond['mean_macro_f1']:>9.3f}{gold_cond['mean_mAP']:>7.3f}"
          f"{gold_cond['mean_baseline_p@1']:>8.3f}")
    print(f"  {'pred-cond':12}{pred_cond['mean_precision@1']:>8.3f}"
          f"{pred_cond['mean_macro_f1']:>9.3f}{pred_cond['mean_mAP']:>7.3f}"
          f"{'':>8}  (coverage "
          f"{sum(v['coverage'] for v in pred_cond['per_aspect'].values()) / 5:.3f})")
    print()
    for a in ASPECTS5:
        g = gold_cond["per_aspect"][a]
        print(f"    {a:12} n={g['n_cells']:6,}  P@1={g['precision@1']:.3f}"
              f"  macroF1={g['macro_f1']:.3f}  mAP={g['mAP']:.3f}"
              f"  (floor {g['baseline_p@1']:.3f})")

    meta = {"arm": args.arm, "use_aug": args.use_aug,
            "class_weights": args.class_weights, "epochs": args.epochs,
            "lr": args.lr, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "best_val_p@1": best, "silver_gold_conditioned": gold_cond,
            "silver_pred_conditioned": pred_cond, "history": history,
            "sub_list": SUB_LIST}
    tag = f"{args.arm}{'_aug' if args.use_aug else ''}{'_w' if args.class_weights else ''}"
    out = MODEL_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pt")
    (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"sub_{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  saved -> {out}")
    return meta


# ---------------------------------------------------------------------------
# Human spot-check (proposal 7.3 + decision (b): oversample coastal_access)
# ---------------------------------------------------------------------------

def sample_check(args) -> None:
    """Draw (evidence span, sub_aspect) pairs for one reviewer to verify.

    Cheap because evidence spans are verbatim and substring-validated (99.6%):
    the reviewer judges a span against a short candidate list, not a whole
    document. Written BLIND - the silver label goes to a separate key file - so
    the exercise measures agreement rather than acquiescence.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rng = random.Random(SEED)

    coastal = con.execute("""
        SELECT review_id, key_aspect, sub_aspect, evidence FROM REVIEW_ASPECTS
        WHERE evidence_valid AND sub_aspect = 'coastal_access'
    """).fetchall()
    rest = con.execute("""
        SELECT review_id, key_aspect, sub_aspect, evidence FROM REVIEW_ASPECTS
        WHERE evidence_valid AND sub_aspect <> 'coastal_access'
          AND sub_aspect IN ({})
    """.format(", ".join(f"'{s}'" for s in SUB_LIST))).fetchall()
    con.close()

    n_c = min(args.coastal_n, len(coastal))
    n_r = max(args.total - n_c, 0)
    picked = rng.sample(coastal, n_c) + rng.sample(rest, min(n_r, len(rest)))
    rng.shuffle(picked)

    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    tasks, key = [], []
    for i, (rid, aspect, sub, ev) in enumerate(picked, 1):
        tasks.append({"sample_id": i, "evidence": ev, "key_aspect": aspect,
                      "candidates": SUB_ASPECT_CHOICES[aspect], "your_choice": ""})
        key.append({"sample_id": i, "review_id": rid, "key_aspect": aspect,
                    "silver_sub_aspect": sub})
    (CHECK_DIR / "sub_check_tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    (CHECK_DIR / "sub_check_key.json").write_text(
        json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")

    import math
    se = math.sqrt(0.8 * 0.2 / n_c)
    print(f"  {len(picked)} cap -> {CHECK_DIR}/sub_check_tasks.json")
    print(f"    coastal_access : {n_c}  (kho tong {len(coastal):,})")
    print(f"    con lai        : {len(picked) - n_c}")
    print(f"  dap an giu rieng -> sub_check_key.json (KHONG mo truoc khi duyet)")
    print(f"  voi n={n_c}, precision cua coastal_access co CI 95% khoang "
          f"+/-{1.96 * se * 100:.0f} diem quanh 0.80")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description="Sub-aspect head (Stage 2).")
    ap.add_argument("--arm", default="no_weights",
                    help="encoder da fine-tune trong models/absa_b/")
    ap.add_argument("--use-aug", action="store_true", help="them split train_aug")
    ap.add_argument("--class-weights", action="store_true",
                    help="inverse-frequency pos_weight (ablation; mac dinh OFF)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-batch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128, help="batch cua encoder")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-check", action="store_true",
                    help="rut bo duyet tay, khong train gi")
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--coastal-n", type=int, default=60)
    args = ap.parse_args()

    if args.sample_check:
        sample_check(args)
        return
    train(args)


if __name__ == "__main__":
    main()
