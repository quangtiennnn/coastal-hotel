# Proposal: Coastal-Distance Word Cloud Banner

## Goal

Produce **two wide banner images** (one Vietnamese, one English) where a single horizontal
strip represents the transition from far-inland hotels to beachfront hotels, with a word cloud
for each distance band occupying a proportional slice of the image.

---

## Visual Design

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                                                                   │
│   ← Band C (≥0.5 km) · 50 % width   │  Band B (0.1–0.5 km) · 30 %  │ Band A ·  │
│                                      │                               │  20 %     │
│   [word cloud — inland topics]        │  [word cloud — near-coast]   │[beach-    │
│                                      │                               │front]     │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
      🌆  city / far inland                                          🏖  beachfront
```

Width proportions mirror real-world distance bands (normalised to the 0 – max-range axis):

| Band | Distance filter | Strip share | Rationale |
|------|----------------|-------------|-----------|
| C    | ≥ 0.5 km        | **50 %**    | Largest review corpus (187 k docs), largest distance span |
| B    | 0.1 – 0.5 km    | **30 %**    | Mid-range corpus (40 k docs), 0.4 km span |
| A    | < 0.1 km        | **20 %**    | Beachfront (24 k docs), tightest range |

---

## Data Sources

All topics come from the pre-fitted BERTopic checkpoints already in `checkpoints/`:

```
checkpoints/
  coast_band_A_en.pkl   coast_band_A_vi.pkl
  coast_band_B_en.pkl   coast_band_B_vi.pkl
  coast_band_C_en.pkl   coast_band_C_vi.pkl
```

Each `.pkl` contains `{"model": BERTopic, "topics": list[int], "language": str}`.

Word-probability pairs are read via:
```python
topics = model.get_topics()          # dict  topic_id → [(word, score), ...]
info   = model.get_topic_info()      # df    Topic, Count, Name, ...
```

---

## Implementation Plan

### Step 1 — Load & aggregate word frequencies per band/language

For each band and each language, merge all non-outlier topics into a single frequency dict,
weighted by topic size:

```python
def band_word_freqs(pkl_path: Path, num_words: int = 40) -> dict[str, float]:
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    info  = model.get_topic_info()

    freq: dict[str, float] = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        weight = int(row["Count"])
        for word, score in (model.get_topic(tid) or [])[:num_words]:
            freq[word] = freq.get(word, 0) + score * weight

    return freq
```

### Step 2 — Build the banner figure

Use a single `matplotlib` figure with three `GridSpec` columns whose widths are
`[50, 30, 20]`:

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud

FIGSIZE   = (24, 6)          # wide landscape — fills a 16:9 slide
BAND_WIDTHS = [50, 30, 20]   # ratios: C, B, A

BAND_COLORS = {
    "C": "#f5e6d3",   # warm sand (inland)
    "B": "#d0eaff",   # soft sky blue (near coast)
    "A": "#006994",   # deep ocean (beachfront)
}
BAND_FONT_COLORS = {
    "C": "darkorange",
    "B": "steelblue",
    "A": "white",
}

def build_banner(lang: str, num_words: int = 60, out_dir: Path = Path("IMG")):
    bands = [
        ("C", Path(f"checkpoints/coast_band_C_{lang}.pkl")),
        ("B", Path(f"checkpoints/coast_band_B_{lang}.pkl")),
        ("A", Path(f"checkpoints/coast_band_A_{lang}.pkl")),
    ]

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs  = gridspec.GridSpec(1, 3, width_ratios=BAND_WIDTHS, wspace=0.01)

    for col_idx, (band_id, pkl_path) in enumerate(bands):
        freq = band_word_freqs(pkl_path, num_words)

        wc = WordCloud(
            width=int(FIGSIZE[0] * 100 * BAND_WIDTHS[col_idx] / 100),
            height=int(FIGSIZE[1] * 100),
            background_color=BAND_COLORS[band_id],
            color_func=lambda *a, **kw: BAND_FONT_COLORS[band_id],
            prefer_horizontal=0.85,
            max_words=num_words,
        ).generate_from_frequencies(freq)

        ax = fig.add_subplot(gs[col_idx])
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        # Band label at the bottom
        label_map = {
            "C": f"Band C  ≥ 0.5 km\n(inland)",
            "B": f"Band B  0.1–0.5 km\n(near coast)",
            "A": f"Band A  < 0.1 km\n(beachfront)",
        }
        ax.set_title(label_map[band_id], fontsize=11, pad=6,
                     color=BAND_FONT_COLORS[band_id])

    fig.suptitle(
        f"Hotel Review Topics by Distance to Coast  ({'Vietnamese' if lang == 'vi' else 'English'})",
        fontsize=14, y=1.02, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"coast_wordcloud_{lang}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out_path}")
```

### Step 3 — Generate both images

```python
build_banner("en")
build_banner("vi")
```

---

## Optional Enhancements

| Enhancement | How |
|---|---|
| Vietnamese font support | Pass `font_path="path/to/NotoSansVI-Regular.ttf"` to `WordCloud` |
| Gradient coastline divider | Draw a vertical `axvline` or use `LinearSegmentedColormap` on a background patch |
| Per-topic sub-rows | Add a second `GridSpec` row showing the top 5 individual topics per band |
| Presentation slide export | Use `plt.savefig(..., dpi=200)` at 1920×540 px to match 16:9 slide aspect |

---

## Output Files

```
IMG/
  coast_wordcloud_en.png    ← English banner
  coast_wordcloud_vi.png    ← Vietnamese banner
```

---

## Where to Add This

Suggest adding as a new notebook `notebooks/15_coast_wordcloud_banner.ipynb`
with sections mirroring this proposal (imports → helper → banner → export).
