"""Generate a pipeline workflow diagram for the coastal-hotel project."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(22, 30))
ax.set_xlim(0, 10)
ax.set_ylim(0, 32)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── colour palette ──────────────────────────────────────────────────────────
C = {
    "title":     "#e2e8f0",
    "phase_bg":  "#1e2130",
    "phase_txt": "#94a3b8",
    "box_data":  "#1a3a5c",   # blue  — data / storage
    "box_proc":  "#1a3d2e",   # green — processing / NLP
    "box_llm":   "#3d2a1a",   # amber — LLM / Claude
    "box_viz":   "#2d1a3d",   # purple— visualisation
    "box_scrape":"#3d1a1a",   # red   — scraping
    "border":    "#334155",
    "arrow":     "#64748b",
    "text":      "#e2e8f0",
    "sub":       "#94a3b8",
    "tag_bg":    "#0f172a",
}

def box(ax, x, y, w, h, label, sublabel="", tag="", color=C["box_proc"]):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor=C["border"],
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(rect)
    cy = y + h / 2
    if tag:
        ax.text(x + 0.18, y + h - 0.22, tag,
                fontsize=6.5, color=C["sub"], va="top", ha="left",
                fontfamily="monospace", zorder=4)
    if sublabel:
        ax.text(x + w / 2, cy + 0.18, label,
                fontsize=9, color=C["text"], va="center", ha="center",
                fontweight="bold", zorder=4)
        ax.text(x + w / 2, cy - 0.22, sublabel,
                fontsize=7.2, color=C["sub"], va="center", ha="center",
                zorder=4)
    else:
        ax.text(x + w / 2, cy, label,
                fontsize=9, color=C["text"], va="center", ha="center",
                fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                        lw=1.6, mutation_scale=14),
        zorder=2,
    )

def phase_label(ax, y, text):
    ax.text(0.18, y, text, fontsize=8, color=C["phase_txt"],
            va="center", fontweight="bold",
            fontfamily="monospace")

def hline(ax, y):
    ax.axhline(y=y, xmin=0.01, xmax=0.99,
               color=C["border"], linewidth=0.7, linestyle="--", zorder=1)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(5, 31.3, "Coastal Hotel Reviews — Full Pipeline",
        fontsize=16, color=C["title"], ha="center", va="center",
        fontweight="bold")
ax.text(5, 30.75, "Vietnamese coastal hotels · Google Maps + Agoda · BERTopic · Claude API",
        fontsize=8.5, color=C["sub"], ha="center", va="center")

# ── Phase 0 : Raw Sources ────────────────────────────────────────────────────
hline(ax, 30.3)
phase_label(ax, 30.0, "PHASE 0 — RAW DATA SOURCES")

box(ax, 0.4, 28.8, 2.8, 0.9, "Google Maps URLs",
    "hotels_processed.csv", "input", C["box_scrape"])
box(ax, 3.5, 28.8, 2.8, 0.9, "Agoda HTML exports",
    "05_agoda_prepare.ipynb", "input", C["box_scrape"])
box(ax, 6.6, 28.8, 2.8, 0.9, "hotel_filtered.csv",
    "metadata source", "input", C["box_scrape"])

# ── Phase 1 : Scraping ───────────────────────────────────────────────────────
hline(ax, 28.5)
phase_label(ax, 28.1, "PHASE 1 — SCRAPING  (Playwright / Chrome)")

box(ax, 0.4, 26.8, 4.2, 0.95, "GMapsReviewsScraper",
    "scraping/get_reviews.py  ·  run.py", "async playwright", C["box_scrape"])
box(ax, 5.4, 26.8, 4.2, 0.95, "Hotel Metadata Scraper",
    "scraping/get_metadata.py  ·  5 concurrent tabs", "async playwright", C["box_scrape"])

# arrows raw→scrape
arrow(ax, 1.8, 28.8, 1.8, 27.75)
arrow(ax, 4.9, 28.8, 4.9, 27.75)   # agoda → different (02 nb)
arrow(ax, 8.0, 28.8, 7.5, 27.75)

box(ax, 1.6, 25.5, 6.8, 0.9, "per-hotel JSON  ·  all_hotels_reviews.json",
    "scraping/outputs/", "raw output", C["box_data"])

arrow(ax, 2.5, 26.8, 2.5, 26.4)
arrow(ax, 7.5, 26.8, 7.5, 26.4)

# ── Phase 2 : Data Preparation ───────────────────────────────────────────────
hline(ax, 25.2)
phase_label(ax, 24.85, "PHASE 2 — DATA PREPARATION")

box(ax, 0.4, 23.6, 2.5, 0.95, "01 data_prepare",
    "normalise hotel list", "nb", C["box_proc"])
box(ax, 3.1, 23.6, 2.5, 0.95, "02 gmaps extract",
    "JSON → DataFrame", "nb", C["box_proc"])
box(ax, 5.8, 23.6, 2.5, 0.95, "03 stay_detail",
    "parse stay info", "nb", C["box_proc"])
box(ax, 0.4, 22.4, 2.5, 0.95, "05 agoda prepare",
    "clean Agoda data", "nb", C["box_proc"])
box(ax, 3.1, 22.4, 2.5, 0.95, "06 gmaps prepare",
    "clean GMaps data", "nb", C["box_proc"])
box(ax, 5.8, 22.4, 2.5, 0.95, "07 merge & preprocess",
    "combine sources", "nb", C["box_proc"])

arrow(ax, 5.0, 25.5, 5.0, 24.55)   # raw JSON down
for xi in [1.65, 4.35, 7.05]:
    arrow(ax, xi, 23.6, xi, 23.35)
for xi in [1.65, 4.35, 7.05]:
    arrow(ax, xi, 22.4, xi, 22.15)
arrow(ax, 5.0, 22.4, 6.5, 22.4)   # merge arrow

# ── Phase 3 : Preprocessing / NLP ────────────────────────────────────────────
hline(ax, 21.9)
phase_label(ax, 21.55, "PHASE 3 — PREPROCESSING & EMBEDDING")

box(ax, 0.4, 20.3, 4.2, 0.95, "04 preprocess  ·  preprocessor.py",
    "ViTokenizer (vi) · spaCy (en)", "NLP", C["box_proc"])
box(ax, 5.2, 20.3, 4.4, 0.95, "embed_to_duckdb.py",
    "paraphrase-multilingual-mpnet-base-v2", "embeddings", C["box_proc"])

arrow(ax, 7.05, 22.4, 7.05, 22.15)
arrow(ax, 7.05, 21.9, 7.05, 21.25)
arrow(ax, 2.5, 21.9, 2.5, 21.25)

box(ax, 1.6, 18.9, 6.8, 1.0, "DuckDB  ·  data/hotel_reviews.db",
    "REVIEWS  ·  EMBEDDINGS  ·  TOPIC_LABELS  ·  TOPIC_ASPECTS", "storage", C["box_data"])
box(ax, 1.6, 17.7, 6.8, 0.85, "Qdrant  ·  qdrant_storage/",
    "local vector store (embedding index)", "vector db", C["box_data"])

arrow(ax, 2.5, 20.3, 2.5, 19.9)
arrow(ax, 7.4, 20.3, 7.4, 19.9)
arrow(ax, 5.0, 19.9, 5.0, 18.55)   # embeddings→qdrant

# ── Phase 4 : Topic Modeling ──────────────────────────────────────────────────
hline(ax, 17.4)
phase_label(ax, 17.05, "PHASE 4 — TOPIC MODELING  (BERTopic)")

box(ax, 0.4, 15.7, 4.2, 1.0, "08 topic_prepare  ·  09 topic_implement",
    "topic_modeling.py · UMAP + HDBSCAN + KeyBERT", "BERTopic", C["box_proc"])
box(ax, 5.2, 15.7, 4.4, 1.0, "10 topic_process  ·  11 coast_time",
    "~1,100 topics · temporal analysis", "topics", C["box_proc"])

arrow(ax, 5.0, 17.7, 5.0, 16.7)
arrow(ax, 2.5, 17.4, 2.5, 16.7)
arrow(ax, 4.6, 16.2, 5.2, 16.2)

box(ax, 1.6, 14.5, 6.8, 0.9, "14 topic_sample_10k  ·  TOPIC_LABELS in DuckDB",
    "sampled topic representation stored", "output", C["box_data"])

arrow(ax, 2.5, 15.7, 2.5, 15.4)
arrow(ax, 7.4, 15.7, 7.4, 15.4)

# ── Phase 5 : Cluster Param Search ───────────────────────────────────────────
hline(ax, 14.2)
phase_label(ax, 13.85, "PHASE 5 — CLUSTER OPTIMISATION")

box(ax, 0.4, 12.65, 4.2, 0.95, "20 cluster_param_search",
    "grid-search HDBSCAN / UMAP hyperparams", "search", C["box_proc"])
box(ax, 5.2, 12.65, 4.4, 0.95, "21 refit_all_runs",
    "refit best config across all data", "refit", C["box_proc"])

arrow(ax, 5.0, 14.5, 5.0, 13.6)
arrow(ax, 4.6, 13.12, 5.2, 13.12)

# ── Phase 6 : LLM Silver Labeling ────────────────────────────────────────────
hline(ax, 12.35)
phase_label(ax, 12.0, "PHASE 6 — LLM SILVER LABELING  (Claude API)")

box(ax, 0.4, 10.7, 2.8, 1.0, "llm_label.py",
    "5-aspect taxonomy schema", "schema", C["box_llm"])
box(ax, 3.4, 10.7, 2.8, 1.0, "llm_label_submit.py",
    "Message Batches API submit", "batch submit", C["box_llm"])
box(ax, 6.4, 10.7, 2.8, 1.0, "llm_label_retrieve.py",
    "poll + write TOPIC_ASPECTS", "batch retrieve", C["box_llm"])

arrow(ax, 5.0, 12.35, 5.0, 11.7)
arrow(ax, 9.5, 12.35, 9.5, 11.7)
arrow(ax, 1.8, 10.7, 1.8, 10.5)
arrow(ax, 4.8, 10.7, 4.8, 10.5)
arrow(ax, 7.8, 10.7, 7.8, 10.5)

box(ax, 1.6, 9.3, 6.8, 0.9,
    "claude-sonnet-4-6  ·  Batch API (50 % off)",
    "facility · amenity · service · experience · loyalty  |  sentiment + weight",
    "LLM", C["box_llm"])

arrow(ax, 5.0, 10.7, 5.0, 10.2)
arrow(ax, 5.0, 9.3, 5.0, 9.0)

box(ax, 1.6, 8.1, 6.8, 0.75, "TOPIC_ASPECTS stored in DuckDB",
    "one row per topic-aspect pair", "output", C["box_data"])

# 17 silver_label_test
box(ax, 7.8, 8.1, 1.8, 0.75, "17 silver test",
    "validation nb", "test", C["box_llm"])
arrow(ax, 5.0, 9.3, 8.7, 8.85)

# ── Phase 7 : Visualisation ───────────────────────────────────────────────────
hline(ax, 7.85)
phase_label(ax, 7.5, "PHASE 7 — ANALYSIS & VISUALISATION")

box(ax, 0.4, 6.2, 2.8, 0.95, "12 sql_implementation",
    "DuckDB queries · analysis", "SQL", C["box_viz"])
box(ax, 3.4, 6.2, 2.8, 0.95, "15 coast_wordcloud",
    "word-cloud banners", "viz", C["box_viz"])
box(ax, 6.4, 6.2, 2.8, 0.95, "16 db_inspection",
    "schema & row counts", "QA", C["box_viz"])

box(ax, 0.4, 4.9, 2.8, 0.95, "18 aspect_visuals",
    "aspect × hotel charts", "viz", C["box_viz"])
box(ax, 3.4, 4.9, 2.8, 0.95, "19 topic_info",
    "topic summaries", "viz", C["box_viz"])
box(ax, 6.4, 4.9, 2.8, 0.95, "22 aspect_visuals_year",
    "temporal aspect trends", "viz ★ active", C["box_viz"])

arrow(ax, 5.0, 8.1, 5.0, 7.85)
for xi in [1.8, 4.8, 7.8]:
    arrow(ax, xi, 7.85, xi, 7.15)
    arrow(ax, xi, 6.2, xi, 5.85)

# ── Phase 8 : Outputs ─────────────────────────────────────────────────────────
hline(ax, 4.6)
phase_label(ax, 4.25, "OUTPUT")

box(ax, 1.0, 3.0, 7.8, 0.95,
    "PNG charts  ·  DuckDB analytics  ·  silver-labeled aspects",
    "aspect sentiment by hotel · temporal trends · topic distributions",
    "final", C["box_data"])

for xi in [1.8, 4.8, 7.8]:
    arrow(ax, xi, 4.9, xi, 3.95)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C["box_scrape"], "Scraping / input"),
    (C["box_proc"],   "Processing / NLP / topic modeling"),
    (C["box_llm"],    "LLM (Claude API)"),
    (C["box_data"],   "Data storage"),
    (C["box_viz"],    "Analysis / visualisation"),
]
lx, ly = 0.4, 2.5
for i, (col, lbl) in enumerate(legend_items):
    rect = mpatches.FancyBboxPatch((lx + i * 1.9, ly), 1.75, 0.45,
                                    boxstyle="round,pad=0.05",
                                    facecolor=col, edgecolor=C["border"],
                                    linewidth=1.0, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + i * 1.9 + 0.875, ly + 0.225, lbl,
            fontsize=6.5, color=C["text"], ha="center", va="center", zorder=4)

plt.tight_layout(pad=0.3)
out = "workflow.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {out}")
