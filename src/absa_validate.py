"""
absa_validate.py
================
Step 1 of ABSA_TRAINING_PROPOSAL.md - "Partition the human gold".

Turns the three Label Studio exports (TripAdvisor_EN, TripAdvisor_VN, BOOKING)
into ONE immutable, reproducible document-level validation set with a fixed
`VALIDATE-dev` / `VALIDATE-test` partition.

Three things happen here, in this order (the order is the point):

  1. **Branding -> experience remap, at load time.** Every *span* annotated
     `Branding` is relabeled `experience` BEFORE the per-aspect majority vote,
     and therefore before the split. Unconditional; no threshold, no branch.
     Because it happens upstream of the split, dev and test can never diverge
     in how the label is treated (proposal, "Branding -> experience: a fixed
     remap").
  2. **6 -> 5 aspects.** facility / amenity / service / experience / loyalty.
     After the remap all six annotated labels are scored - none are dropped.
  3. **35 % dev / 65 % test**, per source, fixed seed, stratified by
     aspect-presence (the 5-bit presence pattern), so both partitions carry
     the rare aspects.

Immutability: the table is written once. Re-running recomputes the assignment
and *verifies* it matches what is stored; any drift is an error unless
--force is passed. The manifest (data/absa_validate_partition.json) records
the spec hash + counts so the split is reproducible outside the DB.

    VALIDATE-dev   all exploratory work, every tuning decision.
    VALIDATE-test  frozen. Read once, at the end. See absa_eval.py, which
                   logs every single read of it to ABSA_TEST_READ_LOG.

Usage:
    uv run python src/absa_validate.py              # build (idempotent)
    uv run python src/absa_validate.py --stats      # report only
    uv run python src/absa_validate.py --force      # allow re-assignment
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter

import duckdb

from absa_en_data import EN_SOURCES, SOURCE_LANG, _is_annotated
from absa_label import DB_PATH, ROOT

# --- the scoring taxonomy: 5 macro aspects, 4 sentiment classes -------------

ASPECTS5 = ["facility", "amenity", "service", "experience", "loyalty"]
CLASSES = ["not_mentioned", "negative", "neutral", "positive"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
ID2CLASS = {i: c for c, i in CLASS2ID.items()}

# Annotator's 6 labels -> the 5 scoring aspects.
# Branding -> experience is the fixed remap; it is applied span-side, at load,
# upstream of the dev/test split.
ANNOT_TO_ASPECT5 = {
    "Facility": "facility",
    "Amenity": "amenity",
    "Service": "service",
    "Experience": "experience",
    "Loyalty": "loyalty",
    "Branding": "experience",  # <- the remap
}
_SENT = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}

# --- partition spec (change any of these and the spec hash changes) ---------

DEV_FRAC = 0.35
SPLIT_SEED = 20260724
SPEC = {
    "dev_frac": DEV_FRAC,
    "seed": SPLIT_SEED,
    "aspects": ASPECTS5,
    "remap": ANNOT_TO_ASPECT5,
    "stratify": "aspect_presence_5bit",
    "sources": sorted(EN_SOURCES),
}
MANIFEST = ROOT / "data" / "absa_validate_partition.json"


def spec_hash() -> str:
    return hashlib.sha256(
        json.dumps(SPEC, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Load: Label Studio doc -> 5 aspect sentiments (Branding folded in)
# ---------------------------------------------------------------------------

def doc_to_labels5(item: dict) -> tuple[str, dict[str, str], int]:
    """One Label Studio item -> (text, {aspect5: sentiment}, n_branding_spans).

    Spans are paired aspect<->sentiment by Label Studio result id, remapped
    6->5 (Branding -> experience), then majority-voted per aspect with ties ->
    neutral (a genuinely mixed aspect). Aspects with no span: not_mentioned.

    The remap being span-side matters: a doc with Branding/positive and
    Experience/negative pools BOTH into the experience vote, rather than
    merging two already-decided document labels.
    """
    text = (item.get("data") or {}).get("text", "") or ""

    spans: dict[str, dict] = {}
    for ann in item.get("annotations", []):
        for res in ann.get("result", []):
            rid = res.get("id")
            labels = (res.get("value") or {}).get("labels") or []
            if rid is None or not labels:
                continue
            span = spans.setdefault(rid, {"aspect": None, "sentiment": None,
                                          "branding": False})
            if res.get("from_name") == "entities":
                if labels[0] in ANNOT_TO_ASPECT5:
                    span["aspect"] = ANNOT_TO_ASPECT5[labels[0]]
                    span["branding"] = labels[0] == "Branding"
            elif res.get("from_name") == "entity_sentiment":
                span["sentiment"] = _SENT.get(labels[0])

    per_aspect: dict[str, list[str]] = {a: [] for a in ASPECTS5}
    n_branding = 0
    for span in spans.values():
        if span["aspect"] and span["sentiment"]:
            per_aspect[span["aspect"]].append(span["sentiment"])
            n_branding += bool(span["branding"])

    labels: dict[str, str] = {}
    for aspect, sents in per_aspect.items():
        if not sents:
            labels[aspect] = "not_mentioned"
            continue
        counts = Counter(sents)
        best = max(counts.values())
        winners = [s for s, n in counts.items() if n == best]
        labels[aspect] = winners[0] if len(winners) == 1 else "neutral"
    return text, labels, n_branding


def load_validate_docs() -> list[dict]:
    """All annotated docs from the three human-annotated sources, remapped.

    doc_id is `{source}_{sha1(text)[:12]}_{occurrence}`, NOT the Label Studio
    `id`: the TripAdvisor_EN export reuses ids (9,990 items share only 3,177
    distinct ids - it is several project exports concatenated), so `id` is not
    a document key there. The Label Studio id is kept in `ls_id` for
    traceability back into the export.
    """
    docs: list[dict] = []
    for source in sorted(EN_SOURCES):
        lang = SOURCE_LANG[source]
        items = json.loads(EN_SOURCES[source].read_text(encoding="utf-8"))
        seen_text: Counter = Counter()
        for item in items:
            if not _is_annotated(item):
                continue  # never labeled: all-not_mentioned would be a lie
            text, labels, n_branding = doc_to_labels5(item)
            if not text.strip():
                continue
            text_hash = hashlib.sha1(text.strip().encode()).hexdigest()[:12]
            occ = seen_text[text_hash]
            seen_text[text_hash] += 1
            docs.append({
                "doc_id": f"{source}_{text_hash}_{occ}",
                "ls_id": str(item.get("id")),
                "text_hash": text_hash,
                "source": source,
                "language": lang,
                "review_text": text,
                "labels": labels,
                "n_branding_spans": n_branding,
            })
    return docs


# ---------------------------------------------------------------------------
# Partition: per source, stratified by aspect-presence, fixed seed
# ---------------------------------------------------------------------------

def presence_key(labels: dict[str, str]) -> str:
    """5-bit aspect-presence pattern, e.g. '10110' - the stratification key."""
    return "".join("1" if labels[a] != "not_mentioned" else "0" for a in ASPECTS5)


def assign_partitions(docs: list[dict]) -> dict[str, str]:
    """{doc_id: 'dev'|'test'} - deterministic given SPEC.

    The unit of assignment is the (source, text_hash) GROUP, not the document:
    131 reviews appear verbatim more than once in the exports, and the same
    text sitting on both sides of the split is a leak, however small.

    Per (source, presence-stratum): sort groups by text hash (stable regardless
    of file order), shuffle with a stratum-specific seeded RNG, take the first
    35% as dev. Any stratum with >= 2 groups contributes at least one group to
    each side, so rare aspect patterns cannot end up entirely in one partition.
    """
    # group -> stratum, decided by the group's first occurrence
    group_stratum: dict[tuple[str, str], str] = {}
    group_docs: dict[tuple[str, str], list[str]] = {}
    for d in docs:
        key = (d["source"], d["text_hash"])
        group_stratum.setdefault(key, presence_key(d["labels"]))
        group_docs.setdefault(key, []).append(d["doc_id"])

    buckets: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, stratum in group_stratum.items():
        buckets.setdefault((key[0], stratum), []).append(key)

    assignment: dict[str, str] = {}
    for (source, stratum), keys in sorted(buckets.items()):
        keys = sorted(keys)
        rng = random.Random(f"{SPLIT_SEED}:{source}:{stratum}")
        rng.shuffle(keys)
        n_dev = round(len(keys) * DEV_FRAC)
        if len(keys) >= 2:
            n_dev = min(max(n_dev, 1), len(keys) - 1)
        for i, key in enumerate(keys):
            side = "dev" if i < n_dev else "test"
            for doc_id in group_docs[key]:
                assignment[doc_id] = side
    return assignment


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

EXPECTED_COLS = (
    ["doc_id", "ls_id", "text_hash", "source", "language", "review_text"]
    + [f"asp5_{a}" for a in ASPECTS5]
    + ["n_branding_spans", "presence_key", "partition", "spec_hash", "built_at"]
)


def _schema_matches(con: duckdb.DuckDBPyConnection) -> bool | None:
    """True/False if ABSA_VALIDATE exists with the/an other schema; None if absent."""
    exists = con.execute("""
        SELECT count(*) FROM information_schema.tables
        WHERE table_name = 'ABSA_VALIDATE'
    """).fetchone()[0]
    if not exists:
        return None
    cols = [r[0] for r in con.execute("DESCRIBE ABSA_VALIDATE").fetchall()]
    return cols == EXPECTED_COLS


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    cols = ",\n            ".join(f"asp5_{a} VARCHAR NOT NULL" for a in ASPECTS5)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS ABSA_VALIDATE (
            doc_id           VARCHAR PRIMARY KEY,
            ls_id            VARCHAR,            -- Label Studio id (NOT unique)
            text_hash        VARCHAR NOT NULL,   -- split unit: duplicates stay together
            source           VARCHAR NOT NULL,
            language         VARCHAR NOT NULL,
            review_text      VARCHAR NOT NULL,
            {cols},
            n_branding_spans INTEGER NOT NULL,
            presence_key     VARCHAR NOT NULL,
            partition        VARCHAR NOT NULL,   -- 'dev' | 'test'
            spec_hash        VARCHAR NOT NULL,
            built_at         TIMESTAMP NOT NULL
        )
    """)


def build(con: duckdb.DuckDBPyConnection, force: bool = False) -> None:
    schema = _schema_matches(con)
    if schema is False:
        n_old = con.execute("SELECT count(*) FROM ABSA_VALIDATE").fetchone()[0]
        if n_old and not force:
            raise SystemExit(
                "ABSA_VALIDATE exists with an older schema and holds "
                f"{n_old:,} rows. Rebuilding re-derives every doc_id, so the "
                "stored partition cannot be verified against the new one. "
                "Pass --force to rebuild from scratch."
            )
        con.execute("DROP TABLE ABSA_VALIDATE")
        schema = None
    ensure_table(con)

    docs = load_validate_docs()
    assignment = assign_partitions(docs)
    print(f"Loaded {len(docs):,} annotated docs from {len(EN_SOURCES)} sources")

    existing = dict(
        con.execute("SELECT doc_id, partition FROM ABSA_VALIDATE").fetchall()
    )
    if existing:
        drift = [d for d, p in existing.items()
                 if assignment.get(d, p) != p]
        missing = [d for d in existing if d not in assignment]
        if (drift or missing) and not force:
            raise SystemExit(
                f"REFUSING to rewrite the partition: {len(drift)} docs would "
                f"change side, {len(missing)} stored docs no longer exist.\n"
                "The dev/test split is meant to be immutable - a doc moving "
                "from test to dev after models were tuned is exactly the leak "
                "the partition exists to prevent. Pass --force only if you "
                "intend to invalidate every result reported so far."
            )
        if drift or missing:
            print(f"--force: reassigning ({len(drift)} moved, {len(missing)} dropped)")
        else:
            print("Existing partition verified: no doc changed side.")

    rows = []
    now_hash = spec_hash()
    for d in docs:
        rows.append((
            d["doc_id"], d["ls_id"], d["text_hash"], d["source"],
            d["language"], d["review_text"],
            *[d["labels"][a] for a in ASPECTS5],
            d["n_branding_spans"], presence_key(d["labels"]),
            assignment[d["doc_id"]], now_hash,
        ))
    con.execute("DELETE FROM ABSA_VALIDATE")
    # doc_id, ls_id, text_hash, source, language, review_text | 5 aspects |
    # n_branding_spans, presence_key, partition, spec_hash  (built_at = now())
    placeholders = ", ".join(["?"] * (10 + len(ASPECTS5)))
    con.executemany(f"""
        INSERT INTO ABSA_VALIDATE
          (doc_id, ls_id, text_hash, source, language, review_text,
           {', '.join('asp5_' + a for a in ASPECTS5)},
           n_branding_spans, presence_key, partition, spec_hash, built_at)
        VALUES ({placeholders}, now())
    """, rows)

    manifest = {
        "spec": SPEC,
        "spec_hash": now_hash,
        "n_docs": len(docs),
        "by_partition": dict(Counter(assignment.values())),
        "by_source_partition": {
            f"{d['source']}/{assignment[d['doc_id']]}": 0 for d in docs
        },
    }
    counts = Counter(f"{d['source']}/{assignment[d['doc_id']]}" for d in docs)
    manifest["by_source_partition"] = dict(sorted(counts.items()))
    manifest["branding_docs"] = {
        s: sum(1 for d in docs if d["source"] == s and d["n_branding_spans"] > 0)
        for s in sorted(EN_SOURCES)
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Wrote ABSA_VALIDATE ({len(rows):,} docs) + manifest -> {MANIFEST}")


# ---------------------------------------------------------------------------
# Read API (used by absa_eval.py / absa_kappa.py / the dev notebook)
# ---------------------------------------------------------------------------

def load_partition(con: duckdb.DuckDBPyConnection, partition: str,
                   language: str | None = None,
                   source: str | None = None) -> list[dict]:
    """Docs of one partition as [{doc_id, language, review_text, labels}, ...]."""
    if partition not in ("dev", "test"):
        raise ValueError("partition must be 'dev' or 'test'")
    where = ["partition = ?"]
    params: list = [partition]
    if language:
        where.append("language = ?")
        params.append(language)
    if source:
        where.append("source = ?")
        params.append(source)
    cols = ", ".join(f"asp5_{a}" for a in ASPECTS5)
    rows = con.execute(f"""
        SELECT doc_id, source, language, review_text, {cols}
        FROM ABSA_VALIDATE WHERE {' AND '.join(where)} ORDER BY doc_id
    """, params).fetchall()
    return [
        {"doc_id": r[0], "source": r[1], "language": r[2], "review_text": r[3],
         "labels": dict(zip(ASPECTS5, r[4:]))}
        for r in rows
    ]


def print_stats(con: duckdb.DuckDBPyConnection) -> None:
    n = con.execute("SELECT count(*) FROM ABSA_VALIDATE").fetchone()[0]
    if not n:
        print("ABSA_VALIDATE is empty - run without --stats first.")
        return
    print(f"\n=== ABSA_VALIDATE ({n:,} docs, spec "
          f"{con.execute('SELECT DISTINCT spec_hash FROM ABSA_VALIDATE').fetchone()[0]}) ===")
    print(f"{'source':16} {'lang':5} {'dev':>8} {'test':>8} {'dev%':>6}")
    for src, lang in con.execute("""
        SELECT DISTINCT source, language FROM ABSA_VALIDATE ORDER BY 1
    """).fetchall():
        d, t = con.execute("""
            SELECT sum(partition='dev'), sum(partition='test')
            FROM ABSA_VALIDATE WHERE source = ?
        """, [src]).fetchone()
        print(f"{src:16} {lang:5} {d:8,} {t:8,} {d/(d+t)*100:5.1f}%")
    d, t = con.execute("""
        SELECT sum(partition='dev'), sum(partition='test') FROM ABSA_VALIDATE
    """).fetchone()
    print(f"{'TOTAL':16} {'':5} {d:8,} {t:8,} {d/(d+t)*100:5.1f}%")

    print("\nAspect presence per partition (must be balanced - that is what "
          "the stratification buys):")
    print(f"{'aspect':12} {'dev present':>12} {'test present':>13} "
          f"{'dev%':>7} {'test%':>7}")
    for a in ASPECTS5:
        dp, dn, tp, tn = con.execute(f"""
            SELECT sum(partition='dev'  AND asp5_{a} <> 'not_mentioned'),
                   sum(partition='dev'),
                   sum(partition='test' AND asp5_{a} <> 'not_mentioned'),
                   sum(partition='test')
            FROM ABSA_VALIDATE
        """).fetchone()
        print(f"{a:12} {dp:12,} {tp:13,} {dp/dn*100:6.1f}% {tp/tn*100:6.1f}%")

    print("\nClass distribution (test partition, per aspect):")
    for a in ASPECTS5:
        counts = dict(con.execute(f"""
            SELECT asp5_{a}, count(*) FROM ABSA_VALIDATE
            WHERE partition='test' GROUP BY 1
        """).fetchall())
        line = "  ".join(f"{c[:3]}={counts.get(c, 0):6,}" for c in CLASSES)
        print(f"  {a:12} {line}")

    print("\nBranding remap (docs with >=1 Branding span, now scored as "
          "experience):")
    for src, n_br, n_all in con.execute("""
        SELECT source, sum(n_branding_spans > 0), count(*)
        FROM ABSA_VALIDATE GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"  {src:16} {n_br:6,} / {n_all:6,}  ({n_br/n_all*100:.1f}%)")


# ---------------------------------------------------------------------------
# The Branding-mismatch follow-up (proposal: "Candidate follow-up")
# ---------------------------------------------------------------------------

# Brand / reputation language, en + vi. Deliberately generic: the point is to
# find out whether brand talk landed in experience/reputation_expectation at
# all, not to build a brand-name recogniser.
_BRAND_WORDS = [
    "brand", "chain", "reputation", "reputable", "famous", "well-known",
    "name", "review", "rating", "star hotel", "expectation", "expected",
    "recommend", "advertis", "photo", "picture",
    "thương hiệu", "nổi tiếng", "danh tiếng", "uy tín", "đánh giá",
    "review", "hình ảnh", "quảng cáo", "kỳ vọng", "mong đợi", "giới thiệu",
]
_NAME_STOP = {"hotel", "resort", "villa", "the", "and", "spa", "beach",
              "khách", "sạn", "nha", "trang", "đà", "nẵng", "phú", "quốc"}


def reputation_check(con: duckdb.DuckDBPyConnection, n: int) -> None:
    """Did brand/reputation language get absorbed into `reputation_expectation`?

    The proposal's stated limitation is that the silver taxonomy has no
    Branding category, while validation now requires brand content to be scored
    as `experience`. It offers one candidate follow-up: sample
    experience/reputation_expectation evidence spans and check how many are
    actually brand mentions. If they are, the mismatch is largely cosmetic; if
    they are not, a future iteration should extend the silver taxonomy rather
    than lean on the remap alone.

    Two independent signals per span: does the evidence name the hotel itself
    (token overlap with REVIEW_DATA.hotel_name, minus generic words), and does
    it use brand/reputation vocabulary.
    """
    rows = con.execute("""
        SELECT ra.evidence, ra.sentiment, r.hotel_name, r.language
        FROM REVIEW_ASPECTS ra
        JOIN REVIEW_DATA r USING (review_id)
        WHERE ra.key_aspect = 'experience'
          AND ra.sub_aspect = 'reputation_expectation'
          AND ra.evidence_valid AND ra.evidence IS NOT NULL
        ORDER BY ra.review_id, ra.aspect_rank
        LIMIT ?
    """, [n]).fetchall()
    if not rows:
        print("\nNo experience/reputation_expectation rows found.")
        return

    n_name = n_word = n_either = 0
    examples: list[tuple[str, str]] = []
    for evidence, _sent, hotel_name, _lang in rows:
        ev = evidence.lower()
        tokens = {t for t in (hotel_name or "").lower().replace("-", " ").split()
                  if len(t) > 3 and t not in _NAME_STOP}
        hit_name = any(t in ev for t in tokens)
        hit_word = any(w in ev for w in _BRAND_WORDS)
        n_name += hit_name
        n_word += hit_word
        n_either += hit_name or hit_word
        if (hit_name or hit_word) and len(examples) < 8:
            examples.append((evidence[:110], "name" if hit_name else "vocab"))

    total = len(rows)
    print(f"\n=== Branding follow-up: experience/reputation_expectation "
          f"({total:,} spans) ===")
    print(f"  names the hotel/brand itself   {n_name:6,}  ({n_name / total:5.1%})")
    print(f"  brand/reputation vocabulary    {n_word:6,}  ({n_word / total:5.1%})")
    print(f"  either signal                  {n_either:6,}  ({n_either / total:5.1%})")
    print("\n  examples:")
    for ev, why in examples:
        print(f"    [{why:4}] {ev}")
    print("\nRead as: a high 'either' share means the silver labeler already had "
          "a home for\nbrand/reputation content inside `experience`, so the "
          "train/validate taxonomy\nmismatch created by the Branding remap is "
          "largely cosmetic. A low share means\nthe silver taxonomy should be "
          "extended in a future iteration instead.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the immutable VALIDATE-dev/test partition."
    )
    ap.add_argument("--stats", action="store_true", help="report only")
    ap.add_argument("--force", action="store_true",
                    help="allow docs to change partition (invalidates results)")
    ap.add_argument("--reputation-check", type=int, nargs="?", const=600,
                    metavar="N",
                    help="sample N reputation_expectation spans and report how "
                         "much brand content the silver already absorbed")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    read_only = args.stats or args.reputation_check is not None
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if args.reputation_check is not None:
        reputation_check(con, args.reputation_check)
        con.close()
        return
    if not args.stats:
        build(con, force=args.force)
    print_stats(con)
    con.close()


if __name__ == "__main__":
    main()
