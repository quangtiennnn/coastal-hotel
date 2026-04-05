# AgenticAI Full Procedure Proposal

## Overview

This proposal outlines a full agentic AI pipeline for the coastal hotel review project, covering:
- **SDK integration** — using the Claude Agent SDK to build review analysis agents
- **Evaluation** — automated quality scoring of agent outputs (silver-label generation)
- **Rerun** — API-driven reprocessing of failed or low-quality runs via `.env`-configured keys

---

## 1. Architecture

```
.env
 └── ANTHROPIC_API_KEY

silver-label/
 ├── proposal.md             ← this file
 ├── agent.py                ← main agentic pipeline
 ├── evaluate.py             ← evaluation & scoring logic
 ├── rerun.py                ← rerun orchestrator for failed/low-score runs
 ├── prompts/
 │   ├── review_analyst.md   ← system prompt for the review analysis agent
 │   └── evaluator.md        ← system prompt for the evaluator agent
 └── outputs/
     ├── runs/               ← raw agent outputs (JSON)
     └── evals/              ← evaluation results (JSON)
```

---

## 2. SDK Integration (`agent.py`)

Use the **Anthropic Python SDK** (`anthropic`) with tool use to build a review analyst agent.

### Agent Responsibilities
- Accept a batch of hotel reviews (from `goorawling/outputs/`)
- Extract structured insights: sentiment, aspects (food, service, atmosphere), rating consistency
- Return a structured silver-label JSON for each review

### Key Design Decisions
| Choice | Rationale |
|---|---|
| `claude-sonnet-4-6` | Best cost/quality balance for batch labeling |
| Tool use (structured output) | Forces consistent JSON schema on every run |
| Async batch processing | Matches existing async scraper architecture |
| `.env` for API key | Keeps credentials out of code and version control |

### `.env` Schema
```dotenv
ANTHROPIC_API_KEY=sk-ant-...
MODEL_ID=claude-sonnet-4-6
MAX_TOKENS=1024
BATCH_CONCURRENCY=5
```

### Pseudocode
```python
# agent.py
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def analyze_review(review: dict) -> dict:
    response = client.messages.create(
        model=os.getenv("MODEL_ID"),
        system=SYSTEM_PROMPT,
        tools=[silver_label_tool],   # enforces output schema
        messages=[{"role": "user", "content": format_review(review)}],
    )
    return extract_tool_use(response)
```

---

## 3. Evaluation (`evaluate.py`)

A second **evaluator agent** scores each silver-label output on defined rubric dimensions.

### Evaluation Rubric
| Dimension | Description | Score |
|---|---|---|
| Accuracy | Sentiment matches star rating | 0–1 |
| Completeness | All aspects (food/service/atmosphere) covered | 0–1 |
| Consistency | Label is internally non-contradictory | 0–1 |
| Faithfulness | No hallucinated content not in source review | 0–1 |

**Composite score** = mean of all dimensions. Threshold for acceptance: **≥ 0.75**.

### Evaluator Flow
```
silver-label output
       ↓
  evaluator agent
       ↓
  eval_result.json  { scores, composite, passed, reason }
       ↓
  passed? → archive to outputs/evals/accepted/
  failed? → queue for rerun
```

### Pseudocode
```python
# evaluate.py
def evaluate(silver_label: dict, original_review: dict) -> EvalResult:
    response = client.messages.create(
        model=os.getenv("MODEL_ID"),
        system=EVALUATOR_PROMPT,
        tools=[eval_tool],
        messages=[{"role": "user", "content": format_eval_input(silver_label, original_review)}],
    )
    scores = extract_tool_use(response)
    composite = mean(scores.values())
    return EvalResult(scores=scores, composite=composite, passed=composite >= 0.75)
```

---

## 4. Rerun Orchestrator (`rerun.py`)

Automatically reprocesses any run that failed evaluation, with configurable retry logic.

### Rerun Strategy
| Trigger | Action |
|---|---|
| `composite < 0.75` | Rerun with higher `temperature` or enriched prompt |
| 3 consecutive failures | Flag for human review, skip |
| API error / timeout | Exponential backoff, retry up to 3× |

### Rerun Flow
```
outputs/evals/          ← scan for failed evals
       ↓
load original review + failed silver label
       ↓
agent.py (re-analyze, optionally with modified prompt)
       ↓
evaluate.py (re-score)
       ↓
passed? → move to accepted/
failed? → increment attempt counter, log to rerun_log.json
```

### Pseudocode
```python
# rerun.py
def rerun_failed(eval_dir: Path, max_attempts: int = 3):
    failed = [f for f in eval_dir.glob("*.json") if not load(f)["passed"]]
    for eval_file in failed:
        attempts = load_attempts(eval_file)
        if attempts >= max_attempts:
            flag_for_human(eval_file)
            continue
        new_label = analyze_review(load_source(eval_file), attempt=attempts)
        new_eval = evaluate(new_label, load_source(eval_file))
        save_result(new_label, new_eval, attempt=attempts + 1)
```

---

## 5. Implementation Plan

### Phase 1 — Foundation
- [ ] Set up `.env` with `ANTHROPIC_API_KEY` and model config
- [ ] Define silver-label JSON schema (output tool definition)
- [ ] Write system prompt for review analyst (`prompts/review_analyst.md`)
- [ ] Implement `agent.py` with async batch processing

### Phase 2 — Evaluation
- [ ] Define evaluator rubric and scoring tool
- [ ] Write evaluator system prompt (`prompts/evaluator.md`)
- [ ] Implement `evaluate.py` and wire to `agent.py` output

### Phase 3 — Rerun & Observability
- [ ] Implement `rerun.py` with retry and backoff logic
- [ ] Add structured logging (`rerun_log.json`, run metadata)
- [ ] End-to-end test with a small batch from `goorawling/outputs/`

### Phase 4 — Integration
- [ ] Connect pipeline to existing hotel review data
- [ ] Add Streamlit UI tab or CLI entry point for the full pipeline
- [ ] Document final silver-label schema in `README.md`

---

## 6. Silver-Label Output Schema

```json
{
  "review_id": "string",
  "source_url": "string",
  "silver_label": {
    "overall_sentiment": "positive | neutral | negative",
    "star_rating_consistent": true,
    "aspects": {
      "food": "positive | neutral | negative | not_mentioned",
      "service": "positive | neutral | negative | not_mentioned",
      "atmosphere": "positive | neutral | negative | not_mentioned",
      "location": "positive | neutral | negative | not_mentioned"
    },
    "key_phrases": ["string"],
    "language": "vi | en | other",
    "confidence": 0.0
  },
  "eval": {
    "composite_score": 0.0,
    "passed": true,
    "attempt": 1
  },
  "metadata": {
    "model": "claude-sonnet-4-6",
    "timestamp": "ISO8601"
  }
}
```

---

## 7. Dependencies

Add to `pyproject.toml`:
```toml
[project.dependencies]
anthropic = ">=0.40.0"
python-dotenv = ">=1.0.0"
```

Install:
```bash
uv add anthropic python-dotenv
```

---

## Notes

- All API keys must live in `.env` — never hardcoded or committed
- Add `.env` to `.gitignore` if not already present
- The evaluator agent must use a separate prompt from the analyst to avoid self-reinforcing bias
- Silver labels are intermediate outputs; human spot-checks are recommended before using as training data
