"""
absa_eval.py
============
Steps 5-7 of ABSA_TRAINING_PROPOSAL.md - everything that scores Model B.

    --predict   run the model over a partition and CACHE its per-sentence
                predictions (the expensive part; do it once per arm)
    --report    score the cache: per-aspect macro-F1 by language, per-class
                P/R/F1 + confusion, with the Cohen's kappa ceiling attached
    --ablation  the pre-specified ablation table across all trained arms,
                including aggregation Rule B as a re-scoring of the baseline
    --gmap-tag  supporting evaluation vs reviewer star tags (same platform)
    --teacher-gap  Model B vs the LLM silver teacher on held-out silver

Two rules this module enforces mechanically rather than by good intentions:

**The test partition is frozen.** `--partition test` requires `--final`, prints
a banner, and appends a row to ABSA_TEST_READ_LOG. The proposal says the test
partition is "touched exactly once, at the end" and that "no decision may be
made after looking at it" - the log is what makes that claim auditable rather
than aspirational. All exploration goes to `--partition dev`.

**Rule B is a re-scoring, not a selection.** Both aggregation rules are defined
in src/absa_model.py before any test read, and `--ablation` always reports both
with their delta. Per the proposal: delta < 3 F1 points => state that results
are robust to the aggregation choice; delta >= 3 => flag it as a limitation.

Because predictions are cached per SENTENCE, changing the aggregation rule
costs no GPU time and cannot silently re-run the model with different settings.

Usage:
    uv run python src/absa_eval.py --predict --arm baseline --partition dev
    uv run python src/absa_eval.py --report  --arm baseline --partition dev
    uv run python src/absa_eval.py --predict --arm baseline --partition test --final
    uv run python src/absa_eval.py --report  --arm baseline --partition test --final
    uv run python src/absa_eval.py --ablation --partition test --final
    uv run python src/absa_eval.py --gmap-tag --arm baseline
    uv run python src/absa_eval.py --teacher-gap --arm baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import duckdb
import torch

from absa_bridge import segment_review
from absa_label import DB_PATH, ROOT
from absa_model import (
    AGGREGATION_RULES,
    MODEL_ROOT,
    aggregate_review,
    load_model,
    predict_sentences,
)
from absa_validate import ASPECTS5, CLASSES, load_partition

RESULTS_DIR = ROOT / "data" / "absa_results"
SMOKE_SEED = 20260727   # only used by the --limit smoke cap


# ---------------------------------------------------------------------------
# The frozen-test guard
# ---------------------------------------------------------------------------

def guard_test_partition(con: duckdb.DuckDBPyConnection, partition: str,
                         final: bool, what: str) -> None:
    if partition != "test":
        return
    if not final:
        raise SystemExit(
            "REFUSING to read VALIDATE-test without --final.\n\n"
            "The test partition is frozen: it is read once, at the end, to "
            "produce the numbers reported in the paper, and no configuration "
            "choice may be made after looking at it. Every exploratory question "
            "belongs on --partition dev.\n\n"
            "If this really is the final scoring pass, re-run with --final."
        )
    con.execute("""
        CREATE TABLE IF NOT EXISTS ABSA_TEST_READ_LOG (
            read_at         TIMESTAMP NOT NULL,
            what            VARCHAR NOT NULL,
            reveals_labels  BOOLEAN NOT NULL,
            argv            VARCHAR NOT NULL
        )
    """)
    # A --predict pass reads the test TEXTS to run an already-fixed model over
    # them; it never touches the gold labels, and step 6 legitimately needs one
    # per ablation arm. Only --report / --ablation reveal the labels, so only
    # those count against the "read exactly once" budget.
    reveals = what != "predict"
    prior = con.execute("""
        SELECT count(*) FROM ABSA_TEST_READ_LOG WHERE reveals_labels
    """).fetchone()[0]
    con.execute("INSERT INTO ABSA_TEST_READ_LOG VALUES (now(), ?, ?, ?)",
                [what, reveals, " ".join(sys.argv)])
    print("=" * 70)
    print("  READING THE FROZEN VALIDATE-test PARTITION")
    if not reveals:
        print("  Prediction pass: test TEXTS only, gold labels untouched.")
        print("  Logged to ABSA_TEST_READ_LOG; does not count as a scoring read.")
    else:
        print(f"  This is scoring read #{prior + 1}, logged to ABSA_TEST_READ_LOG.")
        if prior:
            print("  !! The labels have been read before. The proposal allows")
            print("     exactly one scoring pass; more than one means any")
            print("     configuration chosen in between is no longer test-blind")
            print("     and must be disclosed in the paper.")
        print("  No decision may be made on the basis of what follows.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

def ensure_pred_table(con: duckdb.DuckDBPyConnection) -> None:
    cols = ",\n            ".join(f"pred_{a} VARCHAR NOT NULL" for a in ASPECTS5)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS ABSA_SENT_PRED (
            arm        VARCHAR NOT NULL,
            dataset    VARCHAR NOT NULL,   -- validate_dev / validate_test / gmap_tag / silver_test
            doc_id     VARCHAR NOT NULL,
            sent_idx   INTEGER NOT NULL,
            {cols},
            PRIMARY KEY (arm, dataset, doc_id, sent_idx)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ABSA_PRED_META (
            arm           VARCHAR NOT NULL,
            dataset       VARCHAR NOT NULL,
            predicted_at  TIMESTAMP NOT NULL,
            model_mtime   DOUBLE NOT NULL,   -- so a retrained arm invalidates this
            max_train_cap INTEGER,           -- non-NULL => the model is a smoke run
            n_docs        INTEGER NOT NULL,
            PRIMARY KEY (arm, dataset)
        )
    """)


def _model_mtime(arm: str) -> float:
    path = MODEL_ROOT / arm / "model.pt"
    return path.stat().st_mtime if path.exists() else 0.0


def check_cache_fresh(con: duckdb.DuckDBPyConnection, arm: str,
                      dataset: str) -> None:
    """Warn if the cached predictions predate the model file they came from.

    Retraining an arm and forgetting to re-run --predict would otherwise score
    the OLD model's predictions under the new model's name, silently.
    """
    exists = con.execute("""
        SELECT count(*) FROM information_schema.tables
        WHERE table_name = 'ABSA_PRED_META'
    """).fetchone()[0]
    if not exists:
        return
    row = con.execute("""
        SELECT model_mtime, max_train_cap FROM ABSA_PRED_META
        WHERE arm = ? AND dataset = ?
    """, [arm, dataset]).fetchone()
    if not row:
        return
    cached_mtime, cap = row
    if _model_mtime(arm) > cached_mtime + 1:
        print(f"\n!! STALE CACHE: models/absa_b/{arm}/model.pt is NEWER than "
              f"its cached predictions for {dataset}.\n"
              f"   You are scoring the previous model. Re-run --predict.\n")
    if cap:
        print(f"\n!! SMOKE MODEL: arm '{arm}' was trained with "
              f"--max-train {cap:,}, not on the full training set.\n"
              f"   These figures are a pipeline check, not a result.\n")


def predict_docs(con: duckdb.DuckDBPyConnection, arm: str, dataset: str,
                 docs: list[dict], batch_size: int, max_len: int) -> None:
    """Segment -> predict -> cache. `docs` = [{doc_id, review_text, language}]."""
    ensure_pred_table(con)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(arm, device)
    print(f"arm={arm} encoder={cfg['encoder']} device={device} "
          f"docs={len(docs):,}")

    con.execute("DELETE FROM ABSA_SENT_PRED WHERE arm = ? AND dataset = ?",
                [arm, dataset])

    # Flatten every document into sentences first so the GPU/CPU sees full
    # batches instead of one short document at a time.
    flat: list[tuple[str, int, str]] = []
    for d in docs:
        sents = segment_review(d["review_text"], d.get("language") or "en")
        if not sents:
            sents = [(0, len(d["review_text"]), d["review_text"].strip())]
        for idx, (_, _, sent) in enumerate(sents):
            flat.append((d["doc_id"], idx, sent))
    print(f"  {len(flat):,} sentences "
          f"({len(flat) / max(len(docs), 1):.1f} per doc)")

    t0 = time.time()
    CHUNK = 5000
    for i in range(0, len(flat), CHUNK):
        chunk = flat[i:i + CHUNK]
        preds = predict_sentences(model, tok, [c[2] for c in chunk], device,
                                  batch_size=batch_size, max_len=max_len)
        con.executemany(f"""
            INSERT OR REPLACE INTO ABSA_SENT_PRED
              (arm, dataset, doc_id, sent_idx,
               {', '.join('pred_' + a for a in ASPECTS5)})
            VALUES ({', '.join(['?'] * (4 + len(ASPECTS5)))})
        """, [(arm, dataset, c[0], c[1], *[p[a] for a in ASPECTS5])
              for c, p in zip(chunk, preds)])
        done = min(i + CHUNK, len(flat))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done:,}/{len(flat):,} sentences  ({rate:.0f}/s, "
              f"eta {(len(flat) - done) / max(rate, 1e-9) / 60:.1f} min)",
              flush=True)

    con.execute("""
        INSERT OR REPLACE INTO ABSA_PRED_META
          (arm, dataset, predicted_at, model_mtime, max_train_cap, n_docs)
        VALUES (?, ?, now(), ?, ?, ?)
    """, [arm, dataset, _model_mtime(arm), cfg.get("max_train_cap"), len(docs)])
    print(f"  cached in {time.time() - t0:.0f}s")


def load_cached(con: duckdb.DuckDBPyConnection, arm: str, dataset: str
                ) -> dict[str, dict[str, list[str]]]:
    """{doc_id: {aspect: [per-sentence label, ...]}} from the cache."""
    cols = ", ".join(f"pred_{a}" for a in ASPECTS5)
    rows = con.execute(f"""
        SELECT doc_id, {cols} FROM ABSA_SENT_PRED
        WHERE arm = ? AND dataset = ? ORDER BY doc_id, sent_idx
    """, [arm, dataset]).fetchall()
    out: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        d = out.setdefault(r[0], {a: [] for a in ASPECTS5})
        for k, a in enumerate(ASPECTS5):
            d[a].append(r[1 + k])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Unweighted mean F1 over all FOUR classes, not_mentioned included.

    not_mentioned is kept in the average on purpose: deciding whether an aspect
    is discussed at all is half of what Model B is asked to do, and dropping
    that class would flatter every aspect whose gold is mostly absence
    (loyalty is not_mentioned in 62% of test docs).
    """
    from sklearn.metrics import f1_score
    return f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0)


def score_partition(con: duckdb.DuckDBPyConnection, arm: str, partition: str,
                    rule: str) -> dict:
    """Aggregate cached sentence predictions to documents and score them."""
    dataset = f"validate_{partition}"
    check_cache_fresh(con, arm, dataset)
    cached = load_cached(con, arm, dataset)
    if not cached:
        raise SystemExit(
            f"No cached predictions for arm={arm} on {dataset}.\n"
            f"  uv run python src/absa_eval.py --predict --arm {arm} "
            f"--partition {partition}" + (" --final" if partition == "test" else "")
        )
    everything = load_partition(con, partition)
    docs = [d for d in everything if d["doc_id"] in cached]
    if len(docs) < len(everything):
        # A --limit smoke run leaves a partial cache behind; scoring it would
        # quietly report a number that looks like a full evaluation.
        print(f"\n!! PARTIAL CACHE: {len(docs):,} of {len(everything):,} "
              f"VALIDATE-{partition} docs have predictions.\n"
              f"   These figures are a SMOKE RUN and must not be reported. "
              f"Re-run --predict without --limit.\n")

    results: dict = {"arm": arm, "partition": partition, "rule": rule,
                     "n_docs": len(docs), "by_language": {}, "overall": {}}
    languages = sorted({d["language"] for d in docs})
    for lang in languages + ["all"]:
        subset = [d for d in docs if lang == "all" or d["language"] == lang]
        per_aspect = {}
        for aspect in ASPECTS5:
            gold = [d["labels"][aspect] for d in subset]
            pred = [aggregate_review(cached[d["doc_id"]], aspect, rule)
                    for d in subset]
            per_aspect[aspect] = {
                "macro_f1": macro_f1(gold, pred),
                "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(subset),
                "gold": gold, "pred": pred,
            }
        entry = {"n_docs": len(subset), "per_aspect": per_aspect,
                 "macro_f1": sum(v["macro_f1"] for v in per_aspect.values())
                 / len(per_aspect)}
        if lang == "all":
            results["overall"] = entry
        else:
            results["by_language"][lang] = entry
    return results


def kappa_ceiling(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, str], float]:
    """{(language, aspect): kappa} for the 4-class metric, if it exists yet."""
    exists = con.execute("""
        SELECT count(*) FROM information_schema.tables
        WHERE table_name = 'ABSA_KAPPA'
    """).fetchone()[0]
    if not exists:
        return {}
    return {
        (r[0], r[1]): r[2] for r in con.execute("""
            SELECT language, aspect, kappa FROM ABSA_KAPPA WHERE metric = 'full4'
        """).fetchall()
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(con: duckdb.DuckDBPyConnection, arm: str, partition: str,
           rule: str, detail: bool) -> dict:
    res = score_partition(con, arm, partition, rule)
    kappa = kappa_ceiling(con)

    print(f"\n{'=' * 78}")
    print(f"Model B - arm '{arm}' vs HUMAN ANNOTATION   "
          f"(VALIDATE-{partition}, aggregation: {rule})")
    print(f"{res['n_docs']:,} documents, cross-platform "
          f"(trained on GMap+Agoda silver, scored on TripAdvisor/Booking)")
    print("=" * 78)

    langs = sorted(res["by_language"])
    header = f"{'aspect':12}" + "".join(f"{l:>12}" for l in langs) + f"{'all':>12}"
    if kappa:
        header += f"{'kappa(all)':>12}"
    print("\nPer-aspect macro-F1" + (" (with reliability ceiling)" if kappa else ""))
    print(header)
    for aspect in ASPECTS5:
        line = f"{aspect:12}"
        for lang in langs:
            line += f"{res['by_language'][lang]['per_aspect'][aspect]['macro_f1']:12.3f}"
        line += f"{res['overall']['per_aspect'][aspect]['macro_f1']:12.3f}"
        if kappa:
            k = kappa.get(("all", aspect))
            line += f"{k:12.3f}" if k is not None else f"{'n/a':>12}"
        print(line)
    line = f"{'MEAN':12}"
    for lang in langs:
        line += f"{res['by_language'][lang]['macro_f1']:12.3f}"
    line += f"{res['overall']['macro_f1']:12.3f}"
    print(line)

    if len(langs) == 2:
        a, b = langs
        gap = res["by_language"][a]["macro_f1"] - res["by_language"][b]["macro_f1"]
        print(f"\nBilingual claim (vi ~ en): {a} - {b} = {gap:+.3f} macro-F1")
        print("  The proposal sets no threshold for 'approximately equal', so no")
        print("  verdict is asserted here - report the gap and let it speak. Note")
        print("  that one annotator labeled BOTH languages, so any per-language")
        print("  difference in their behaviour is confounded with this number.")

    if not kappa:
        print("\n!! No Cohen's kappa on record. Per the proposal's annotation-")
        print("   provenance section, every F1 above is PROVISIONAL until the")
        print("   double-annotation subset is scored: these numbers measure")
        print("   agreement with ONE annotator's judgement, not with a")
        print("   validated consensus standard.  ->  src/absa_kappa.py")
    else:
        print("\nkappa is the reliability ceiling: a model cannot meaningfully be")
        print("credited with F1 above the level at which two humans agree.")

    if detail:
        from sklearn.metrics import classification_report, confusion_matrix
        for aspect in ASPECTS5:
            d = res["overall"]["per_aspect"][aspect]
            print(f"\n--- {aspect} - per-class (neg/neu are the hard cases) ---")
            print(classification_report(d["gold"], d["pred"], labels=CLASSES,
                                        target_names=CLASSES, zero_division=0))
            cm = confusion_matrix(d["gold"], d["pred"], labels=CLASSES)
            print(f"confusion (rows=gold, cols=pred): {CLASSES}")
            for name, row in zip(CLASSES, cm):
                print(f"  {name:14}" + "".join(f"{v:8,}" for v in row))

    _persist(con, res)
    return res


def _persist(con: duckdb.DuckDBPyConnection, res: dict) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ABSA_EVAL_RESULTS (
            arm       VARCHAR NOT NULL,
            partition VARCHAR NOT NULL,
            rule      VARCHAR NOT NULL,
            language  VARCHAR NOT NULL,
            aspect    VARCHAR NOT NULL,
            macro_f1  DOUBLE,
            accuracy  DOUBLE,
            n_docs    INTEGER,
            scored_at TIMESTAMP NOT NULL,
            PRIMARY KEY (arm, partition, rule, language, aspect)
        )
    """)
    rows = []
    buckets = dict(res["by_language"])
    buckets["all"] = res["overall"]
    for lang, entry in buckets.items():
        for aspect, d in entry["per_aspect"].items():
            rows.append((res["arm"], res["partition"], res["rule"], lang, aspect,
                         d["macro_f1"], d["accuracy"], entry["n_docs"]))
        rows.append((res["arm"], res["partition"], res["rule"], lang, "MEAN",
                     entry["macro_f1"], None, entry["n_docs"]))
    con.executemany("""
        INSERT OR REPLACE INTO ABSA_EVAL_RESULTS
          (arm, partition, rule, language, aspect, macro_f1, accuracy, n_docs,
           scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())
    """, rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in res.items() if k not in ("by_language", "overall")}
    slim["by_language"] = {
        lang: {"n_docs": e["n_docs"], "macro_f1": e["macro_f1"],
               "per_aspect": {a: {"macro_f1": d["macro_f1"],
                                  "accuracy": d["accuracy"]}
                              for a, d in e["per_aspect"].items()}}
        for lang, e in {**res["by_language"], "all": res["overall"]}.items()
    }
    path = RESULTS_DIR / f"{res['arm']}_{res['partition']}_{res['rule']}.json"
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ablation table (proposal: all pre-specified, all reported)
# ---------------------------------------------------------------------------

ABLATION_ROWS = [
    ("baseline", "primary", "1 Baseline - silver only, class weights ON"),
    ("augmented", "primary", "2 Augmented - silver + rare-cell top-up"),
    ("no_weights", "primary", "3 Baseline, class weights OFF"),
    ("frozen", "primary", "4 Frozen encoder + linear heads"),
    ("baseline", "ruleB", "5 Aggregation Rule B (any-negative), arm 1 re-scored"),
]


def ablation(con: duckdb.DuckDBPyConnection, partition: str) -> None:
    print(f"\n{'=' * 78}\nABLATION TABLE - VALIDATE-{partition}\n"
          f"Every arm is pre-specified and always reported; none is gated on a\n"
          f"qualitative judgement.\n{'=' * 78}")

    scored, missing = {}, []
    for arm, rule, label in ABLATION_ROWS:
        try:
            scored[(arm, rule)] = score_partition(con, arm, partition, rule)
        except SystemExit:
            missing.append((arm, rule, label))

    if not scored:
        raise SystemExit("No arm has cached predictions yet - run --predict first.")

    langs = sorted(next(iter(scored.values()))["by_language"])
    print(f"\n{'row':52}" + "".join(f"{l:>10}" for l in langs) + f"{'all':>10}")
    for arm, rule, label in ABLATION_ROWS:
        res = scored.get((arm, rule))
        if res is None:
            print(f"{label:52}" + "".join(f"{'--':>10}" for _ in langs) + f"{'--':>10}")
            continue
        line = f"{label:52}"
        for lang in langs:
            line += f"{res['by_language'][lang]['macro_f1']:10.3f}"
        line += f"{res['overall']['macro_f1']:10.3f}"
        print(line)

    # Row 5 is a sensitivity, not a selection: both rules were fixed before the
    # test partition was read, so this delta is reported, never chosen between.
    base = scored.get(("baseline", "primary"))
    ruleb = scored.get(("baseline", "ruleB"))
    if base and ruleb:
        delta = (ruleb["overall"]["macro_f1"] - base["overall"]["macro_f1"]) * 100
        print(f"\nAggregation sensitivity: Rule B - primary = {delta:+.2f} F1 points")
        if abs(delta) < 3:
            print("  |delta| < 3  ->  results are ROBUST to the choice of "
                  "aggregation rule.\n  State this explicitly in the paper.")
        else:
            print("  |delta| >= 3  ->  LIMITATION: the reported figure is "
                  "sensitive to a post-hoc\n  heuristic. The choice must be "
                  "justified rather than assumed.")
        print("\n  per-aspect delta (Rule B - primary, F1 points):")
        for aspect in ASPECTS5:
            d = (ruleb["overall"]["per_aspect"][aspect]["macro_f1"]
                 - base["overall"]["per_aspect"][aspect]["macro_f1"]) * 100
            print(f"    {aspect:12} {d:+7.2f}")

    if missing:
        print("\nNot scored (train the arm, then --predict it):")
        for arm, rule, label in missing:
            print(f"  {label}   ->  --arm {arm}")

    for (arm, rule), res in scored.items():
        _persist(con, res)


# ---------------------------------------------------------------------------
# Supporting evaluation 1: reviewer star tags (same platform)
# ---------------------------------------------------------------------------

def gmap_tag_eval(con: duckdb.DuckDBPyConnection, arm: str, n: int,
                  batch_size: int, max_len: int, rule: str) -> None:
    """P/R/F1 vs the 1-5-star reviewer tags for facility / service / amenity.

    Same platform as training, so this cannot prove cross-platform
    generalization - what it does is confirm the model is not merely
    reproducing the LLM teacher, since the star tags come from reviewers, not
    from the teacher.

    Held-out by construction: every review in ABSA_SAMPLE (any split, including
    train_aug) is excluded, so nothing here was ever seen by the labeler or the
    model.
    """
    tag_aspects = ["facility", "service", "amenity"]
    docs = [
        {"doc_id": r[0], "review_text": r[1], "language": r[2]}
        for r in con.execute("""
            SELECT DISTINCT r.review_id, r.review_text, r.language
            FROM GOLD_REVIEW_ASPECTS g
            JOIN REVIEW_DATA r ON r.review_id = g.review_id
            WHERE g.gold_source = 'gmap_tag'
              AND r.review_text IS NOT NULL AND length(r.review_text) >= 30
              AND r.review_id NOT IN (SELECT review_id FROM ABSA_SAMPLE)
            ORDER BY r.review_id
            LIMIT ?
        """, [n]).fetchall()
    ]
    if not docs:
        raise SystemExit("No held-out gmap_tag reviews found.")

    check_cache_fresh(con, arm, "gmap_tag")
    cached = load_cached(con, arm, "gmap_tag")
    if set(d["doc_id"] for d in docs) - set(cached):
        predict_docs(con, arm, "gmap_tag", docs, batch_size, max_len)
        cached = load_cached(con, arm, "gmap_tag")

    gold_rows = con.execute("""
        SELECT review_id, key_aspect, sentiment FROM GOLD_REVIEW_ASPECTS
        WHERE gold_source = 'gmap_tag'
    """).fetchall()
    gold: dict[str, dict[str, str]] = {}
    for review_id, aspect, sentiment in gold_rows:
        gold.setdefault(review_id, {})[aspect] = sentiment

    from sklearn.metrics import classification_report

    print(f"\n{'=' * 78}\nSUPPORTING - arm '{arm}' vs gmap_tag reviewer stars "
          f"({len(docs):,} held-out reviews)\n"
          f"Same platform as training; reviewer-generated, so it is independent "
          f"of the LLM teacher.\n{'=' * 78}")
    for aspect in tag_aspects:
        pairs = [(gold[d["doc_id"]][aspect],
                  aggregate_review(cached[d["doc_id"]], aspect, rule))
                 for d in docs
                 if d["doc_id"] in cached and aspect in gold.get(d["doc_id"], {})]
        if not pairs:
            continue
        g = [p[0] for p in pairs]
        # The star tag asserts the aspect IS discussed; not_mentioned is the
        # model declining to predict it, kept visible rather than dropped.
        p = [p[1] for p in pairs]
        print(f"\n--- {aspect}  (n={len(pairs):,}) ---")
        print(classification_report(g, p, labels=CLASSES, target_names=CLASSES,
                                    zero_division=0))
        covered = sum(1 for x in p if x != "not_mentioned") / len(p)
        agree = sum(1 for a, b in zip(g, p) if a == b) / len(p)
        print(f"  coverage (model predicts the aspect at all): {covered:.1%}")
        print(f"  exact agreement with the star tag:           {agree:.1%}")


# ---------------------------------------------------------------------------
# Supporting evaluation 2: the teacher gap
# ---------------------------------------------------------------------------

def teacher_gap(con: duckdb.DuckDBPyConnection, arm: str, batch_size: int,
                max_len: int, rule: str) -> None:
    """Model B vs the LLM silver teacher, on the silver's own held-out split.

    Measured on split='test' - reviews the teacher labeled but the model never
    trained on. Agreement on the training split would be a memorisation check,
    not a teacher gap.

    The proposal is emphatic that this number means nothing on its own:

      small gap + strong human-gold F1  -> the model learned genuine aspect-
                                           sentiment signal
      small gap + weak   human-gold F1  -> the model overfit the teacher's own
                                           labeling biases
      large gap                         -> a training problem (optimization,
                                           capacity, bridge label noise),
                                           not a data problem

    So always read it next to the VALIDATE-test table from --report.
    """
    docs = [
        {"doc_id": r[0], "review_text": r[1], "language": r[2]}
        for r in con.execute("""
            SELECT s.review_id, r.review_text, r.language
            FROM ABSA_SAMPLE s
            JOIN REVIEW_DATA r USING (review_id)
            JOIN REVIEW_ASPECT_ROLLUP ro USING (review_id)
            WHERE s.split = 'test' AND r.review_text IS NOT NULL
            ORDER BY s.review_id
        """).fetchall()
    ]
    if not docs:
        raise SystemExit("No labeled silver test reviews found.")

    check_cache_fresh(con, arm, "silver_test")
    cached = load_cached(con, arm, "silver_test")
    if set(d["doc_id"] for d in docs) - set(cached):
        predict_docs(con, arm, "silver_test", docs, batch_size, max_len)
        cached = load_cached(con, arm, "silver_test")

    cols = ", ".join(f"asp5_{a}" for a in ASPECTS5)
    teacher = {
        r[0]: {a: (r[1 + i] or "not_mentioned") for i, a in enumerate(ASPECTS5)}
        for r in con.execute(
            f"SELECT review_id, {cols} FROM REVIEW_ASPECT_ROLLUP").fetchall()
    }

    print(f"\n{'=' * 78}\nTEACHER GAP - arm '{arm}' vs the LLM silver teacher\n"
          f"{len(docs):,} held-out silver reviews (split='test'; never trained on)\n"
          f"{'=' * 78}")
    print(f"\n{'aspect':12}{'agreement':>12}{'macro-F1':>12}")
    f1s = []
    for aspect in ASPECTS5:
        pairs = [(teacher[d["doc_id"]][aspect],
                  aggregate_review(cached[d["doc_id"]], aspect, rule))
                 for d in docs if d["doc_id"] in cached and d["doc_id"] in teacher]
        g, p = [x[0] for x in pairs], [x[1] for x in pairs]
        f1 = macro_f1(g, p)
        f1s.append(f1)
        agree = sum(a == b for a, b in zip(g, p)) / len(pairs)
        print(f"{aspect:12}{agree:12.1%}{f1:12.3f}")
    print(f"{'MEAN':12}{'':12}{sum(f1s) / len(f1s):12.3f}")
    print("\nInterpret ONLY together with the human-gold table (--report):")
    print("  small gap + strong human-gold F1 -> genuine signal learned")
    print("  small gap + weak   human-gold F1 -> the teacher's biases absorbed")
    print("  large gap                        -> a training problem, not data")


# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Evaluate Model B.")
    ap.add_argument("--arm", default="baseline",
                    help="trained arm in models/absa_b/")
    ap.add_argument("--partition", default="dev", choices=["dev", "test"])
    ap.add_argument("--rule", default="primary", choices=AGGREGATION_RULES)
    ap.add_argument("--final", action="store_true",
                    help="required to read the frozen VALIDATE-test partition")
    ap.add_argument("--predict", action="store_true", help="fill the prediction cache")
    ap.add_argument("--report", action="store_true", help="score the cache")
    ap.add_argument("--detail", action="store_true",
                    help="per-class report + confusion per aspect")
    ap.add_argument("--ablation", action="store_true", help="full ablation table")
    ap.add_argument("--gmap-tag", action="store_true", help="reviewer-star support eval")
    ap.add_argument("--teacher-gap", action="store_true", help="vs the LLM teacher")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap documents (smoke runs only - never for reported numbers)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-gmap", type=int, default=2000,
                    help="held-out gmap_tag reviews to score")
    args = ap.parse_args()

    if not any([args.predict, args.report, args.ablation, args.gmap_tag,
                args.teacher_gap]):
        ap.error("pick at least one of --predict / --report / --ablation / "
                 "--gmap-tag / --teacher-gap")

    con = duckdb.connect(str(DB_PATH))
    try:
        if args.predict or args.report or args.ablation:
            guard_test_partition(
                con, args.partition, args.final,
                "predict" if args.predict else
                ("ablation" if args.ablation else "report"))

        if args.predict:
            docs = load_partition(con, args.partition)
            if args.limit and args.limit < len(docs):
                # Fixed-seed SAMPLE, not a head slice: docs come out ordered by
                # doc_id, which is grouped by source, so the first N would all
                # be Booking/en and the smoke run would never exercise vi.
                docs = random.Random(SMOKE_SEED).sample(docs, args.limit)
                print(f"--limit: {len(docs):,} docs (SMOKE RUN - not reportable)")
            predict_docs(con, args.arm, f"validate_{args.partition}", docs,
                         args.batch, args.max_len)
        if args.report:
            report(con, args.arm, args.partition, args.rule, args.detail)
        if args.ablation:
            ablation(con, args.partition)
        if args.gmap_tag:
            gmap_tag_eval(con, args.arm, args.n_gmap, args.batch, args.max_len,
                          args.rule)
        if args.teacher_gap:
            teacher_gap(con, args.arm, args.batch, args.max_len, args.rule)
    finally:
        con.close()


if __name__ == "__main__":
    main()
