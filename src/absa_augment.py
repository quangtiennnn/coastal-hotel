"""
absa_augment.py
===============
Step 3 of ABSA_TRAINING_PROPOSAL.md - the rare-cell top-up.

Model B trains on the silver labels, so thin silver (aspect x sentiment) cells
directly cap what it can learn: neutral is ~3% of all rows, and negative is a
small minority for amenity, experience and loyalty. The proposal makes the
top-up **unconditional** - it is not gated on "if recall looks weak", it is
built because the **Augmented** arm of the ablation table requires it.

What this does (selection only - it spends no API money):

  1. Count the real (key_aspect x sentiment) cells in REVIEW_ASPECTS and work
     out each cell's deficit against the largest cell of its aspect.
  2. For each thin cell, take the reviews that already carry that label as
     SEEDS and average their REVIEW_EMBEDDINGS into a cell centroid.
  3. Mine the unlabeled coastal reviews (everything not already in
     ABSA_SAMPLE) for candidates that (a) hit that aspect's keyword lexicon in
     the review's language and (b) hit a sentiment-cue lexicon for negative /
     neutral cells, then rank what survives by cosine similarity to the cell
     centroid.
  4. Write the winners into ABSA_SAMPLE with **split='train_aug'**.

Guard-rails (proposal): train split only; provenance tag preserved via
`run_id='train_aug:<aspect>:<sentiment>'`; VALIDATE-dev / VALIDATE-test and
every gold set are untouched - they live in other tables entirely, and the
candidate pool excludes every review_id already in ABSA_SAMPLE.

After this runs, the EXISTING labeling pipeline picks the rows up unchanged:
`absa_label_submit.py` sends any ABSA_SAMPLE review with no REVIEW_ASPECTS
rows, and `absa_bridge.py` then bridges them into SENTENCE_LABELS.

Usage:
    uv run python src/absa_augment.py --dry-run     # show the allocation
    uv run python src/absa_augment.py               # write 2,500 rows
    uv run python src/absa_augment.py --n 3000
    uv run python src/absa_augment.py --stats
"""

from __future__ import annotations

import argparse

import duckdb

from absa_label import DB_PATH
from absa_validate import ASPECTS5

DEFAULT_N = 2500
SUBSAMPLE_SEED = 20260728
MIN_CHARS = 30          # same floor as absa_sample.py
CANDIDATE_POOL = 400    # candidates scored per cell before the top-k cut
SEED_CAP = 2000         # seeds averaged into a cell centroid

# Cells to top up. Neutral everywhere (it is ~3% of the corpus and the main
# confusion source), plus negative for the aspects where negatives are thin.
TARGET_SENTIMENTS = ["neutral", "negative"]

# --- keyword lexicons ------------------------------------------------------
# Deliberately high-recall and low-precision: they are a *filter* in front of
# the embedding ranker, not a classifier. Precision comes from the centroid
# similarity; these only keep the candidate pool on-aspect.

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "facility": [
        # en
        "room", "bed", "bathroom", "shower", "toilet", "pool", "aircon",
        "air conditioning", "clean", "dirty", "smell", "furniture", "tv",
        "wifi", "elevator", "lift", "balcony", "window", "noise", "broken",
        # vi
        "phòng", "giường", "nhà tắm", "phòng tắm", "vòi sen", "toilet",
        "hồ bơi", "bể bơi", "máy lạnh", "điều hòa", "sạch", "bẩn", "mùi",
        "nội thất", "tivi", "wifi", "thang máy", "ban công", "cửa sổ", "hỏng",
    ],
    "amenity": [
        "location", "beach", "sea", "center", "centre", "near", "far",
        "walk", "restaurant", "market", "airport", "parking", "breakfast",
        "buffet", "gym", "spa", "bar", "shuttle",
        "vị trí", "biển", "bãi biển", "trung tâm", "gần", "xa", "đi bộ",
        "nhà hàng", "chợ", "sân bay", "bãi đỗ", "bãi xe", "ăn sáng",
        "buffet", "phòng gym", "spa", "quán bar", "xe đưa đón",
    ],
    "service": [
        "staff", "service", "reception", "check in", "check-in", "check out",
        "checkout", "friendly", "rude", "helpful", "slow", "wait", "manager",
        "housekeeping", "booking", "request",
        "nhân viên", "phục vụ", "dịch vụ", "lễ tân", "nhận phòng", "trả phòng",
        "thân thiện", "nhiệt tình", "thái độ", "chậm", "chờ", "quản lý",
        "dọn phòng", "đặt phòng", "yêu cầu",
    ],
    "experience": [
        "price", "value", "worth", "expensive", "cheap", "quiet", "view",
        "atmosphere", "comfortable", "relax", "romantic", "luxury", "money",
        "overall", "disappointed", "expectation",
        "giá", "đáng tiền", "đắt", "rẻ", "yên tĩnh", "ồn", "view", "cảnh",
        "không gian", "thoải mái", "thư giãn", "lãng mạn", "sang trọng",
        "tổng thể", "thất vọng", "kỳ vọng", "xứng đáng",
    ],
    "loyalty": [
        "return", "come back", "again", "recommend", "next time", "revisit",
        "never again", "would not", "wouldn't",
        "quay lại", "trở lại", "lần sau", "giới thiệu", "sẽ đến", "lần nữa",
        "không bao giờ", "sẽ không",
    ],
}

SENTIMENT_KEYWORDS: dict[str, list[str]] = {
    "negative": [
        "not", "no", "bad", "poor", "worst", "disappoint", "terrible",
        "awful", "rude", "dirty", "broken", "smell", "complain", "problem",
        "but", "however", "unfortunately", "old", "noisy", "slow", "expensive",
        "không", "chưa", "tệ", "kém", "dở", "thất vọng", "bẩn", "hỏng",
        "mùi", "phàn nàn", "vấn đề", "nhưng", "tuy nhiên", "tiếc", "cũ",
        "ồn", "chậm", "đắt", "quá tệ",
    ],
    "neutral": [
        "ok", "okay", "average", "fine", "normal", "acceptable", "so so",
        "so-so", "nothing special", "as expected", "standard", "basic",
        "just", "enough", "could be better",
        "ổn", "bình thường", "tạm", "tạm được", "trung bình", "chấp nhận",
        "không có gì đặc biệt", "như mong đợi", "cơ bản", "tàm tạm", "đủ",
    ],
}


def _like_clause(col: str, words: list[str]) -> str:
    """OR of case-insensitive substring matches (works for vi + en alike)."""
    parts = [f"lower({col}) LIKE '%' || lower({_q(w)}) || '%'" for w in words]
    return "(" + " OR ".join(parts) + ")"


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def cell_counts(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, str], int]:
    return {
        (a, s): n for a, s, n in con.execute("""
            SELECT key_aspect, sentiment, count(*) FROM REVIEW_ASPECTS
            WHERE key_aspect <> 'other' AND sentiment IS NOT NULL
            GROUP BY 1, 2
        """).fetchall()
    }


def allocate(counts: dict[tuple[str, str], int], n_total: int
             ) -> dict[tuple[str, str], int]:
    """Split the budget across thin cells in proportion to their deficit.

    A cell's deficit is how far it sits below the largest cell of the same
    aspect (which is always the positive one). Deficit-proportional rather
    than equal allocation, so the emptiest cells - loyalty/neutral at 10 rows,
    amenity/neutral at 283 - get the most help.
    """
    deficits: dict[tuple[str, str], float] = {}
    for aspect in ASPECTS5:
        biggest = max((counts.get((aspect, s), 0)
                       for s in ("positive", "negative", "neutral")), default=0)
        for sentiment in TARGET_SENTIMENTS:
            have = counts.get((aspect, sentiment), 0)
            if have < biggest:
                deficits[(aspect, sentiment)] = biggest - have

    total = sum(deficits.values())
    if not total:
        return {}
    quotas = {c: int(d / total * n_total) for c, d in deficits.items()}
    # largest-remainder top-up so the budget is spent exactly
    short = n_total - sum(quotas.values())
    order = sorted(deficits, key=lambda c: -(deficits[c] / total * n_total
                                             - quotas[c]))
    for c in order[:short]:
        quotas[c] += 1
    return {c: q for c, q in quotas.items() if q > 0}


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def centroid_sql(con: duckdb.DuckDBPyConnection, aspect: str,
                 sentiment: str, language: str | None = None) -> str | None:
    """Mean embedding of the reviews already labeled (aspect, sentiment).

    Averaged in Python and inlined as an array literal: DuckDB has no
    aggregate over FLOAT[768], and a 768-float literal is ~10 KB of SQL - far
    cheaper than pulling 152k candidate vectors into the client to score them.

    `language` restricts the SEEDS to one language, which is the whole point of
    the language-aware path. The first version of this miner built ONE centroid
    from all seeds; because the labeled pool is 65% Vietnamese, that centroid
    sat in the Vietnamese region of the multilingual embedding space and scored
    every Vietnamese candidate ~0.067 higher REGARDLESS OF CONTENT. Taking the
    top ~2% then turned a small systematic offset into near-total exclusion:
    2,480 vi vs 20 en, from a candidate pool that was 55% English. Comparing
    only within a language cancels that offset exactly.
    """
    where_lang = "AND r.language = ?" if language else ""
    params: list = [aspect, sentiment]
    if language:
        params.append(language)
    params.append(SEED_CAP)
    seeds = con.execute(f"""
        SELECT e.embedding
        FROM REVIEW_ASPECTS ra
        JOIN REVIEW_EMBEDDINGS e ON e.review_id = ra.review_id
        JOIN REVIEW_DATA r ON r.review_id = ra.review_id
        WHERE ra.key_aspect = ? AND ra.sentiment = ? AND ra.evidence_valid
          {where_lang}
        LIMIT ?
    """, params).fetchall()
    if not seeds:
        return None
    dim = len(seeds[0][0])
    acc = [0.0] * dim
    for (vec,) in seeds:
        for i, v in enumerate(vec):
            acc[i] += v
    n = len(seeds)
    return "[" + ", ".join(f"{v / n:.6f}" for v in acc) + f"]::FLOAT[{dim}]"


def mine_cell(con: duckdb.DuckDBPyConnection, aspect: str, sentiment: str,
              quota: int, exclude: set[str],
              language: str | None = None) -> list[tuple]:
    """Top-`quota` unlabeled reviews for one (aspect, sentiment[, language]) cell.

    When `language` is given, both the centroid and the candidates are
    restricted to it, so ranking never compares across languages.
    """
    cent = centroid_sql(con, aspect, sentiment, language)
    if cent is None:
        print(f"  {aspect}/{sentiment}: no seed rows - skipped")
        return []

    kw_aspect = _like_clause("r.review_text", ASPECT_KEYWORDS[aspect])
    kw_sent = _like_clause("r.review_text", SENTIMENT_KEYWORDS[sentiment])
    not_in = ""
    if exclude:
        vals = ", ".join(_q(x) for x in exclude)
        not_in = f"AND r.review_id NOT IN ({vals})"
    lang_filter = f"AND r.language = {_q(language)}" if language else ""

    rows = con.execute(f"""
        SELECT r.review_id, r.source, r.language, r.review_text,
               array_cosine_similarity(e.embedding, {cent}) AS sim
        FROM REVIEW_DATA r
        JOIN REVIEW_EMBEDDINGS e ON e.review_id = r.review_id
        WHERE r.review_text IS NOT NULL
          AND length(r.review_text) >= {MIN_CHARS}
          AND r.review_id NOT IN (SELECT review_id FROM ABSA_SAMPLE)
          {not_in}
          {lang_filter}
          AND {kw_aspect}
          AND {kw_sent}
        ORDER BY sim DESC
        LIMIT {max(quota, CANDIDATE_POOL)}
    """).fetchall()
    return [(r[0], r[1], r[2], r[3], float(r[4])) for r in rows[:quota]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def mine_to_sentence_target(con: duckdb.DuckDBPyConnection, language: str,
                            target_sentences: int, dry_run: bool) -> None:
    """Mine one language until it contributes ~`target_sentences` sentences.

    The unit that matters is the SENTENCE, not the review: Model B trains on
    sentences, and mined reviews are far longer than average (en 6.1, vi 4.3
    sentences vs 3.3 / 2.6 in the base sample) because a candidate must match
    both an aspect keyword and a sentiment cue, which long reviews do more
    often. Sizing this in reviews would therefore overshoot badly.

    Sentence counts are computed by running the SAME segmenter the bridge uses,
    which reproduces the eventual SENTENCE_LABELS count exactly (verified
    1.0000 on both languages) - so the target is hit before any money is spent.
    """
    from absa_bridge import segment_review

    counts = cell_counts(con)
    # Same deficit-proportional shape as the main allocator, just used as
    # relative weights; the absolute size is set by the sentence target.
    weights = allocate(counts, 10_000)
    total_w = sum(weights.values())

    print(f"Mining {language} to a target of {target_sentences:,} sentences\n")
    print(f"  {'cell':22} {'reviews':>8} {'sentences':>10} {'target':>8}")
    picked: list[tuple] = []
    seen: set[str] = set()
    n_sent = 0
    # Each cell gets its OWN sentence sub-target, proportional to the same
    # deficit weights. Filling cells greedily in weight order instead would
    # exhaust the budget on the biggest cells and leave facility and loyalty
    # with no English at all - the complement has to mirror the Vietnamese
    # cell distribution, not just its largest entries.
    for (aspect, sentiment), w in sorted(weights.items()):
        sub_target = target_sentences * w / total_w
        # over-select: mined reviews average 4-6 sentences, so ask for plenty
        quota = max(int(sub_target / 2), 30)
        rows = mine_cell(con, aspect, sentiment, quota, seen, language)
        taken, cell_sent = 0, 0
        for review_id, source, lang, text, sim in rows:
            if cell_sent >= sub_target:
                break
            n = max(len(segment_review(text, lang or language)), 1)
            seen.add(review_id)
            picked.append((review_id, f"train_aug:{aspect}:{sentiment}",
                           source, lang))
            cell_sent += n
            taken += 1
        n_sent += cell_sent
        print(f"  {aspect + '/' + sentiment:22} {taken:8,} {cell_sent:10,} "
              f"{sub_target:8.0f}")

    print(f"\nSelected {len(picked):,} {language} reviews -> "
          f"{n_sent:,} sentences ({n_sent / max(len(picked),1):.2f} per review)")
    if dry_run:
        print("--dry-run: nothing written.")
        return

    con.executemany("""
        INSERT INTO ABSA_SAMPLE
          (review_id, run_id, topic_id, source, language, split, sampled_at)
        VALUES (?, ?, -1, ?, ?, 'train_aug', now())
    """, picked)
    print(f"Inserted {len(picked):,} rows into ABSA_SAMPLE (split='train_aug').")
    print("Next: uv run python src/absa_label_submit.py")


def subsample_language(con: duckdb.DuckDBPyConnection, language: str,
                       target_sentences: int, dry_run: bool) -> None:
    """Hold part of one language's top-up OUT of training, per cell.

    Why this exists: the top-up must stay at the weight the proposal specified
    (~2-3k reviews, here 24.2% of training sentences). Restoring the language
    balance by ADDING English instead would push it to ~35%, and augmentation
    is a form of oversampling - the same pressure class weighting applies. We
    measured what that does to this model: arm 1 vs arm 3, -5.0 F1, through
    over-prediction of `negative` and of aspect presence. Growing the top-up
    risks reproducing that pathology through a different channel.

    Excluded reviews are NOT deleted - they are already labeled and paid for.
    Their split becomes 'train_aug_excess' in both ABSA_SAMPLE and
    SENTENCE_LABELS, so training skips them, the exclusion is inspectable, and
    a future "does MORE augmentation help" arm can pick them straight back up.

    Selection is a fixed-seed RANDOM subsample within each cell, deliberately
    not "keep the highest-cosine ones": keeping only the best-ranked reviews
    would make the Vietnamese half systematically more extreme than the English
    half, which is the asymmetry this whole exercise exists to remove.
    """
    import random as _random

    rows = con.execute("""
        SELECT s.run_id, s.review_id,
               count(DISTINCT sl.review_id || '#' || sl.sent_idx) AS n_sent
        FROM ABSA_SAMPLE s JOIN SENTENCE_LABELS sl USING (review_id)
        WHERE s.split = 'train_aug' AND s.language = ?
        GROUP BY 1, 2 ORDER BY 1, 2
    """, [language]).fetchall()
    if not rows:
        print(f"No train_aug rows for {language}.")
        return

    by_cell: dict[str, list[tuple[str, int]]] = {}
    for run_id, review_id, n_sent in rows:
        by_cell.setdefault(run_id, []).append((review_id, n_sent))
    total = sum(n for cell in by_cell.values() for _, n in cell)
    keep_frac = target_sentences / total

    print(f"{language}: {total:,} sentences -> target {target_sentences:,} "
          f"(keep {keep_frac:.1%}, per cell)\n")
    print(f"  {'cell':34} {'keep':>6} {'excess':>7} {'sent kept':>10}")
    excess: list[str] = []
    kept_sent = 0
    for run_id, members in sorted(by_cell.items()):
        members = sorted(members)
        _random.Random(f"{SUBSAMPLE_SEED}:{run_id}").shuffle(members)
        cell_target = sum(n for _, n in members) * keep_frac
        acc, keep_n = 0, 0
        for review_id, n_sent in members:
            if acc >= cell_target:
                excess.append(review_id)
            else:
                acc += n_sent
                keep_n += 1
        kept_sent += acc
        print(f"  {run_id:34} {keep_n:6,} {len(members) - keep_n:7,} {acc:10,}")

    print(f"\n  kept {kept_sent:,} sentences, held out {len(excess):,} reviews")
    if dry_run:
        print("--dry-run: nothing written.")
        return
    ids = [(r,) for r in excess]
    con.executemany(
        "UPDATE ABSA_SAMPLE SET split='train_aug_excess' WHERE review_id = ?", ids)
    con.executemany(
        "UPDATE SENTENCE_LABELS SET split='train_aug_excess' WHERE review_id = ?", ids)
    print(f"Moved {len(excess):,} reviews to split='train_aug_excess' "
          f"(labeled, retained, simply not trained on).")


def run(con: duckdb.DuckDBPyConnection, n_total: int, dry_run: bool) -> None:
    counts = cell_counts(con)
    quotas = allocate(counts, n_total)

    print(f"Silver cell counts and top-up allocation (budget {n_total:,}):\n")
    print(f"  {'cell':24} {'have':>8} {'top-up':>8}")
    for aspect in ASPECTS5:
        for sentiment in ("positive",) + tuple(TARGET_SENTIMENTS):
            have = counts.get((aspect, sentiment), 0)
            q = quotas.get((aspect, sentiment), 0)
            flag = f"{q:8,}" if q else ("       -" if sentiment == "positive"
                                        else "       0")
            print(f"  {aspect + '/' + sentiment:24} {have:8,} {flag}")
    print()

    if dry_run:
        print("--dry-run: nothing written. Drop the flag to mine + insert.")
        return

    picked: list[tuple] = []
    seen: set[str] = set()
    for (aspect, sentiment), quota in sorted(quotas.items()):
        rows = mine_cell(con, aspect, sentiment, quota, seen)
        for review_id, source, language, sim in rows:
            seen.add(review_id)
            picked.append((review_id, f"train_aug:{aspect}:{sentiment}",
                           source, language, sim))
        sims = [r[3] for r in rows]
        if rows:
            print(f"  {aspect}/{sentiment:9} mined {len(rows):5,}  "
                  f"cos {min(sims):.3f}-{max(sims):.3f}")

    if not picked:
        print("Nothing mined.")
        return

    # topic_id = -1: these rows come from cell mining, not from a topic cell
    # (the column is NOT NULL, so -1 is the sentinel for "no topic").
    con.executemany("""
        INSERT INTO ABSA_SAMPLE
          (review_id, run_id, topic_id, source, language, split, sampled_at)
        VALUES (?, ?, -1, ?, ?, 'train_aug', now())
    """, [(p[0], p[1], p[2], p[3]) for p in picked])

    print(f"\nInserted {len(picked):,} rows into ABSA_SAMPLE (split='train_aug').")
    print("They are now visible to the existing labeling pipeline:")
    print("  uv run python src/absa_label_submit.py --dry-run   # inspect")
    print("  uv run python src/absa_label_submit.py             # YOU submit")
    print("  uv run python src/absa_label_retrieve.py --wait")
    print("  uv run python src/absa_bridge.py                   # -> SENTENCE_LABELS")


def print_stats(con: duckdb.DuckDBPyConnection) -> None:
    print("\n=== ABSA_SAMPLE by split ===")
    for split, n in con.execute("""
        SELECT split, count(*) FROM ABSA_SAMPLE GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"  {split:12} {n:7,}")
    n_aug, n_lab = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE review_id IN
                 (SELECT review_id FROM REVIEW_ASPECTS))
        FROM ABSA_SAMPLE WHERE split = 'train_aug'
    """).fetchone()
    if n_aug:
        print(f"\ntrain_aug: {n_lab:,}/{n_aug:,} labeled "
              f"({n_aug - n_lab:,} still awaiting the LLM pass)")
        print("  by target cell:")
        for run_id, n in con.execute("""
            SELECT run_id, count(*) FROM ABSA_SAMPLE
            WHERE split = 'train_aug' GROUP BY 1 ORDER BY 1
        """).fetchall():
            print(f"    {run_id:34} {n:6,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rare-cell top-up for Model B.")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"reviews to mine (default {DEFAULT_N})")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the allocation, write nothing")
    ap.add_argument("--stats", action="store_true", help="report only")
    ap.add_argument("--language", choices=["en", "vi"],
                    help="mine ONE language, ranking within it (language-aware)")
    ap.add_argument("--target-sentences", type=int,
                    help="with --language: mine until this many sentences")
    ap.add_argument("--subsample", action="store_true",
                    help="with --language + --target-sentences: hold the "
                         "EXCESS out of training as split='train_aug_excess' "
                         "instead of mining more")
    args = ap.parse_args()

    read_only = args.stats or args.dry_run
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if args.subsample:
        if not (args.language and args.target_sentences):
            ap.error("--subsample needs --language and --target-sentences")
        subsample_language(con, args.language, args.target_sentences,
                           args.dry_run)
        con.close()
        return
    if args.language and args.target_sentences:
        mine_to_sentence_target(con, args.language, args.target_sentences,
                                args.dry_run)
        print_stats(con)
        con.close()
        return
    if not args.stats:
        run(con, args.n, args.dry_run)
    print_stats(con)
    con.close()


if __name__ == "__main__":
    main()
