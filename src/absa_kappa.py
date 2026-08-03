"""
absa_kappa.py
=============
Step 2 of ABSA_TRAINING_PROPOSAL.md - the **required** double-annotation
subset and its Cohen's kappa.

Why this exists (proposal, "Annotation provenance"): all 34,468 human-annotated
documents carry exactly ONE annotation from ONE annotator id. Every F1 in the
paper therefore measures agreement with one individual's judgement, not with a
validated consensus standard. The fix committed to in the proposal is to
double-annotate a subset and report kappa as the **reliability ceiling** - a
model cannot meaningfully be credited with F1 above the level at which two
humans agree.

This module does the two machine-side halves. The middle step is human.

    --export   draw the subset from VALIDATE-test and write a Label Studio
               import file for the second annotator. BLIND: the file carries
               the raw text only - no original labels, no Model B predictions.
    --score    read the second annotator's export back, recompute both
               annotators' document labels through the SAME 6->5 +
               Branding->experience path, and report kappa per aspect and per
               language into ABSA_KAPPA.

The subset is drawn from VALIDATE-test (not dev) on purpose: the measured
agreement has to characterise the partition the paper actually reports on.
Drawing it does NOT count as "reading" the test partition - no label is
inspected and no modelling decision follows from it.

Usage:
    uv run python src/absa_kappa.py --export                  # 250/language
    uv run python src/absa_kappa.py --export --n-per-lang 300
    uv run python src/absa_kappa.py --score data/raw/kappa_round2.json
    uv run python src/absa_kappa.py --stats
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import duckdb

from absa_label import DB_PATH, ROOT
from absa_validate import ASPECTS5, CLASSES, doc_to_labels5, presence_key

KAPPA_SEED = 20260724
EXPORT_DIR = ROOT / "data" / "kappa"


# ---------------------------------------------------------------------------
# Export: draw the subset, write the blind annotation task
# ---------------------------------------------------------------------------

def draw_subset(con: duckdb.DuckDBPyConnection, n_per_lang: int) -> list[dict]:
    """Stratified random draw from VALIDATE-test, n per language.

    Stratified by the same 5-bit aspect-presence key as the partition itself,
    proportionally, so the subset's aspect mix mirrors the test partition -
    otherwise kappa for a rare aspect would rest on a handful of documents.
    """
    rows = con.execute("""
        SELECT doc_id, source, language, review_text, presence_key
        FROM ABSA_VALIDATE WHERE partition = 'test' ORDER BY doc_id
    """).fetchall()
    if not rows:
        raise SystemExit("ABSA_VALIDATE is empty - run src/absa_validate.py first.")

    by_lang: dict[str, list] = {}
    for r in rows:
        by_lang.setdefault(r[2], []).append(r)

    picked: list[dict] = []
    for lang, lrows in sorted(by_lang.items()):
        strata: dict[str, list] = {}
        for r in lrows:
            strata.setdefault(r[4], []).append(r)
        # proportional allocation, largest-remainder so the total is exact
        quotas = {k: len(v) / len(lrows) * n_per_lang for k, v in strata.items()}
        base = {k: int(q) for k, q in quotas.items()}
        short = n_per_lang - sum(base.values())
        for k, _ in sorted(quotas.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:short]:
            base[k] += 1

        for stratum, srows in sorted(strata.items()):
            take = min(base.get(stratum, 0), len(srows))
            if not take:
                continue
            rng = random.Random(f"{KAPPA_SEED}:{lang}:{stratum}")
            for r in rng.sample(sorted(srows), take):
                picked.append({"doc_id": r[0], "source": r[1], "language": r[2],
                               "review_text": r[3], "presence_key": r[4]})
    return picked


def export(con: duckdb.DuckDBPyConnection, n_per_lang: int, force: bool) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ABSA_KAPPA_SUBSET (
            doc_id     VARCHAR PRIMARY KEY,
            language   VARCHAR NOT NULL,
            source     VARCHAR NOT NULL,
            drawn_at   TIMESTAMP NOT NULL
        )
    """)
    n_have = con.execute("SELECT count(*) FROM ABSA_KAPPA_SUBSET").fetchone()[0]
    if n_have and not force:
        raise SystemExit(
            f"ABSA_KAPPA_SUBSET already holds {n_have:,} docs. Re-drawing after "
            "annotation has started would silently change what kappa describes; "
            "pass --force if that is really what you want."
        )

    picked = draw_subset(con, n_per_lang)
    con.execute("DELETE FROM ABSA_KAPPA_SUBSET")
    con.executemany(
        "INSERT INTO ABSA_KAPPA_SUBSET (doc_id, language, source, drawn_at) "
        "VALUES (?, ?, ?, now())",
        [(d["doc_id"], d["language"], d["source"]) for d in picked],
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in sorted({d["language"] for d in picked}):
        docs = [d for d in picked if d["language"] == lang]
        # Label Studio import format. `data` carries the text plus the doc_id
        # so --score can join back. NOTHING ELSE: the second annotator must not
        # see the first annotator's labels or any model output.
        task = [{"data": {"text": d["review_text"], "doc_id": d["doc_id"],
                          "language": d["language"], "source": d["source"]}}
                for d in docs]
        path = EXPORT_DIR / f"kappa_round2_{lang}.json"
        path.write_text(json.dumps(task, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"  {lang}: {len(task):3,} docs -> {path}")

    print(f"\nDrew {len(picked):,} docs from VALIDATE-test "
          f"({n_per_lang}/language), recorded in ABSA_KAPPA_SUBSET.")
    print("Hand these files to a SECOND annotator, blind, using the original "
          "guidelines (6 labels incl. Branding + entity_sentiment).")
    print("Then: uv run python src/absa_kappa.py --score <their export>.json")


# ---------------------------------------------------------------------------
# Score: Cohen's kappa, annotator 1 vs annotator 2
# ---------------------------------------------------------------------------

def load_round2(paths: list) -> dict[str, dict[str, str]]:
    """{doc_id: {aspect5: sentiment}} from the second annotator's export.

    Runs through the identical doc_to_labels5() as the reference set, so the
    Branding->experience remap and the majority-vote-with-neutral-ties rule are
    the same on both sides - kappa measures annotator disagreement, not two
    different aggregation conventions.
    """
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            data = item.get("data") or {}
            doc_id = data.get("doc_id")
            if not doc_id:
                continue
            _, labels, _ = doc_to_labels5(item)
            out[doc_id] = labels
    return out


def cohen_kappa(a: list[str], b: list[str], classes: list[str]) -> float:
    """Cohen's kappa without a sklearn dependency (identical formula)."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    exp = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in classes)
    if exp == 1.0:
        return float("nan")  # both annotators constant on one class
    return (obs - exp) / (1 - exp)


def score(con: duckdb.DuckDBPyConnection, paths: list) -> None:
    round2 = load_round2(paths)
    if not round2:
        raise SystemExit("No documents with a `doc_id` found in the export(s).")

    cols = ", ".join(f"asp5_{a}" for a in ASPECTS5)
    ref = {
        r[0]: {"language": r[1], "labels": dict(zip(ASPECTS5, r[2:]))}
        for r in con.execute(f"""
            SELECT doc_id, language, {cols} FROM ABSA_VALIDATE
            WHERE doc_id IN (SELECT doc_id FROM ABSA_KAPPA_SUBSET)
        """).fetchall()
    }
    shared = sorted(set(ref) & set(round2))
    missing = len(ref) - len(shared)
    print(f"Matched {len(shared):,} doubly-annotated docs "
          f"({missing:,} of the drawn subset not returned)")
    if not shared:
        raise SystemExit("No overlap with ABSA_KAPPA_SUBSET - wrong export?")

    con.execute("""
        CREATE TABLE IF NOT EXISTS ABSA_KAPPA (
            language    VARCHAR NOT NULL,
            aspect      VARCHAR NOT NULL,
            metric      VARCHAR NOT NULL,  -- full4 / presence / sentiment
            kappa       DOUBLE,
            n_docs      INTEGER NOT NULL,
            pct_agree   DOUBLE,
            scored_at   TIMESTAMP NOT NULL,
            PRIMARY KEY (language, aspect, metric)
        )
    """)
    con.execute("DELETE FROM ABSA_KAPPA")

    langs = sorted({ref[d]["language"] for d in shared})
    out_rows = []
    for lang in langs + ["all"]:
        docs = [d for d in shared if lang == "all" or ref[d]["language"] == lang]
        for aspect in ASPECTS5:
            a1 = [ref[d]["labels"][aspect] for d in docs]
            a2 = [round2[d].get(aspect, "not_mentioned") for d in docs]

            # 1. full 4-class agreement - the headline reliability ceiling
            k4 = cohen_kappa(a1, a2, CLASSES)
            agree = sum(x == y for x, y in zip(a1, a2)) / len(docs) if docs else 0
            out_rows.append((lang, aspect, "full4", k4, len(docs), agree))

            # 2. presence only: do they even agree the aspect is discussed?
            p1 = ["y" if x != "not_mentioned" else "n" for x in a1]
            p2 = ["y" if x != "not_mentioned" else "n" for x in a2]
            out_rows.append((lang, aspect, "presence",
                             cohen_kappa(p1, p2, ["y", "n"]), len(docs),
                             sum(x == y for x, y in zip(p1, p2)) / len(docs)))

            # 3. sentiment, restricted to docs BOTH called present
            both = [(x, y) for x, y in zip(a1, a2)
                    if x != "not_mentioned" and y != "not_mentioned"]
            if both:
                s1, s2 = [x for x, _ in both], [y for _, y in both]
                out_rows.append((lang, aspect, "sentiment",
                                 cohen_kappa(s1, s2, CLASSES[1:]), len(both),
                                 sum(x == y for x, y in zip(s1, s2)) / len(both)))
            else:
                out_rows.append((lang, aspect, "sentiment", None, 0, None))

    con.executemany("""
        INSERT OR REPLACE INTO ABSA_KAPPA
          (language, aspect, metric, kappa, n_docs, pct_agree, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, now())
    """, out_rows)
    print_stats(con)


def print_stats(con: duckdb.DuckDBPyConnection) -> None:
    exists = con.execute("""
        SELECT count(*) FROM information_schema.tables WHERE table_name='ABSA_KAPPA'
    """).fetchone()[0]
    if not exists or not con.execute("SELECT count(*) FROM ABSA_KAPPA").fetchone()[0]:
        print("\nABSA_KAPPA is empty - the double-annotation round has not been "
              "scored yet.\nEvery F1 reported before it exists is PROVISIONAL "
              "(proposal, 'Annotation provenance').")
        return

    print("\n=== Cohen's kappa - inter-annotator reliability ceiling ===")
    for metric, title in [("full4", "4-class (not_mentioned/neg/neu/pos)"),
                          ("presence", "aspect presence only"),
                          ("sentiment", "sentiment | both say present")]:
        print(f"\n{title}")
        langs = [r[0] for r in con.execute(
            "SELECT DISTINCT language FROM ABSA_KAPPA ORDER BY 1").fetchall()]
        print(f"  {'aspect':12} " + " ".join(f"{l:>14}" for l in langs))
        for aspect in ASPECTS5:
            cells = []
            for lang in langs:
                r = con.execute("""
                    SELECT kappa, n_docs FROM ABSA_KAPPA
                    WHERE language=? AND aspect=? AND metric=?
                """, [lang, aspect, metric]).fetchone()
                cells.append("           n/a" if not r or r[0] is None
                             else f"{r[0]:8.3f} (n={r[1]})".rjust(14))
            print(f"  {aspect:12} " + " ".join(cells))
    print("\nRead as: a model's F1 on an aspect is not meaningfully credible "
          "above that aspect's kappa.\nLandis-Koch: <0.20 slight, 0.21-0.40 "
          "fair, 0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Double-annotation kappa subset.")
    ap.add_argument("--export", action="store_true",
                    help="draw the subset + write the blind annotation files")
    ap.add_argument("--n-per-lang", type=int, default=250,
                    help="docs per language (proposal: 200-300)")
    ap.add_argument("--score", nargs="+", metavar="EXPORT.json",
                    help="score the second annotator's Label Studio export(s)")
    ap.add_argument("--stats", action="store_true", help="print kappa table")
    ap.add_argument("--force", action="store_true", help="allow re-drawing")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Only --export and --score write; anything else keeps the DB unlocked.
    con = duckdb.connect(str(DB_PATH), read_only=not (args.export or args.score))
    if args.export:
        export(con, args.n_per_lang, args.force)
    if args.score:
        score(con, [Path(p) for p in args.score])
    if args.stats or not (args.export or args.score):
        print_stats(con)
    con.close()


if __name__ == "__main__":
    main()
