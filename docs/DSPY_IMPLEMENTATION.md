# DSPy Implementation Guide

This document describes the specific DSPy implementation in this project for optimizing taxonomy-to-content matching prompts.

## Overview

Our implementation provides two optimization paths:
1. **Quick optimization** (`optimize_prompts`): Uses `BootstrapFewShot` with existing database matching results
2. **Dataset-based optimization** (`optimize-dataset`): Uses provided dataset files with GEPA or BootstrapFewShotWithRandomSearch

## Dataset Format

### CSV Format

Required columns:
- `taxonomy_category`: Category of the taxonomy page
- `taxonomy_description`: Description of the taxonomy page
- `content_summaries`: Formatted list of content pages with index, title, URL, and preview
- `best_match_index`: Index of the best matching content page (integer)

Optional columns:
- `taxonomy_keywords`: Comma-separated keywords (defaults to "None" if missing)
- `reasoning`: Explanation text (defaults to empty string if missing)

Example:
```csv
taxonomy_category,taxonomy_description,taxonomy_keywords,content_summaries,best_match_index,reasoning
Technology,Tech content,tech innovation,"0. Title: Tech Post
   URL: https://example.com/tech
   Preview: Content about technology...",0,"Good semantic match"
```

### JSON Format

Array of objects with the same fields as CSV columns:

```json
[
  {
    "taxonomy_category": "Technology",
    "taxonomy_description": "Tech content",
    "taxonomy_keywords": "tech, innovation",
    "content_summaries": "0. Title: Tech Post\n   URL: https://example.com/tech\n   Preview: Content...",
    "best_match_index": 0,
    "reasoning": "Good match"
  }
]
```

## Configuration

DSPy optimization settings in `.env`:

```bash
# GEPA optimization budget (light/medium/heavy)
DSPY_OPTIMIZATION_BUDGET=medium

# Training set split ratio (0.2 recommended for prompt optimizers)
DSPY_TRAIN_SPLIT_RATIO=0.2

# Random seed for reproducibility
DSPY_OPTIMIZATION_SEED=42

# Reflection model for GEPA (defaults to LLM_MODEL if empty)
DSPY_REFLECTION_MODEL=

# Parallel evaluation threads (1 for debugging, 4-24 for production)
DSPY_NUM_THREADS=1

# Number of example results to display in evaluation (0 disables)
DSPY_DISPLAY_TABLE=5

# Explicit GEPA budget (alternative to auto budget)
# DSPY_MAX_FULL_EVALS=10
# DSPY_MAX_METRIC_CALLS=1000
```

## CLI Commands

### Quick Optimization

```bash
python -m src.cli optimize_prompts [--num-trials N]
```

Uses existing matching results from the database. Fast but limited by available data.

### Dataset-Based Optimization

```bash
python -m src.cli optimize-dataset \
  --dataset path/to/dataset.csv \
  --optimizer gepa \
  --budget medium
```

**Options:**
- `--dataset`: Path to CSV or JSON dataset file (required)
- `--optimizer`: `gepa`, `bootstrap-random-search`, or `bootstrap` (default: `gepa`)
- `--budget`: GEPA budget level (`light`/`medium`/`heavy`) - ignored if `--max-full-evals` or `--max-metric-calls` provided
- `--max-full-evals`: Explicit GEPA budget: maximum full evaluations
- `--max-metric-calls`: Explicit GEPA budget: maximum metric calls
- `--train-split`: Training set split ratio (default: 0.2)
- `--output`: Path to save optimized model (default: `prompt-optimiser/models/matcher_vN.json`)
- `--num-threads`: Parallel evaluation threads (default: 1)
- `--display-table`: Number of example results to display (default: 5, 0 disables)
- `--seed`: Random seed for reproducibility

**Budget Guidelines (GEPA):**
- `light`: ~500-1600 metric calls, 6-12 full evaluations (quick experimentation)
- `medium`: Moderate metric calls and generations (balanced optimization)
- `heavy`: Comprehensive search with higher cost (thorough optimization)

**Note**: Dataset-based optimization is expensive by nature due to multiple iterations, metric evaluations, and LLM calls. The expense is inherent to the optimization process.

## Implementation Details

### DSPyOptimizer Class

Located in `src/optimization/dspy_optimizer.py`.

**Key Methods:**

- `load_training_dataset(dataset_path)`: Loads examples from CSV or JSON file
- `optimize_with_gepa(...)`: GEPA optimization with configurable budget
- `optimize_with_dataset(...)`: Dataset-based optimization supporting multiple optimizer types
- `optimize(...)`: Quick BootstrapFewShot optimization (existing method)
- `predict_match(taxonomy, content_items) -> (index, rubric)`: Selector returns best index; rubric hints are not used for gating
- `judge_candidate(taxonomy, candidate) -> rubric`: Judge scores a single candidate with rubric fields used for deterministic gating

### Model Persistence

Optimized models are saved using DSPy's built-in persistence into `prompt-optimiser/models/`:
- `matcher_vN.json` – Versioned model artifacts (N auto-incremented per run)

Models are automatically loaded by `CategorizationService` via `matcher_latest.json`. Use `scripts/promote_optimized_model.py` to copy the latest versioned artifact (`matcher_vN.json`) to `matcher_latest.json` once you’ve tested it locally. Because everything runs on localhost (no staging tier), promotion is the safeguard between “experiment” and “in use”—keep prior `matcher_vN.json` files so you can re-promote if you need to roll back.

### Training Data Preparation

The `prepare_training_data()` method converts database matching results into DSPy examples:
- Filters for successful matches (`LLM_CATEGORIZED` or `SEMANTIC_MATCHED`)
- Formats content summaries with index, title, URL, and preview
- Creates examples with proper input field marking

### Evaluation Metrics

The `accuracy_metric()` method:
- Checks exact match on `best_match_index`
- Returns score between 0 and 1

### Selector vs. Judge

- Selector (DSPy): Chooses `best_match_index` from candidate summaries. It may emit reasoning/rubric-style hints, but these are not trusted for acceptance.
- Judge (rubric scorer): Independently scores the selector’s chosen candidate, returning:
  - `topic_alignment`, `intent_fit`, `entity_overlap`, `temporal_relevance`, `decision`, and `reasoning`.
- The workflow feeds the judge’s rubric into `_accept_by_rubric()` for deterministic gating. Selector output alone never triggers acceptance.

## Best Practices

1. **Dataset Size**: Recommend at least 50 examples, optimal for 300+ examples
2. **Train/Validation Split**: Use 20% training, 80% validation for prompt optimizers
3. **Budget Selection**: Start with `light` for experimentation, use `medium` or `heavy` for production
4. **Parallel Evaluation**: Use `--num-threads 4-24` for production runs (start with 1 for debugging)
5. **Reproducibility**: Always set `--seed` for consistent results across runs
6. **Model Saving**: Use JSON format for human inspection, pickle for complex optimizations

## Integration with Matching Workflow

The optimized model is automatically used by `CategorizationService` when available:
- Scans `prompt-optimiser/models/` for `matcher_v*.json` on initialization
- Falls back to unoptimized model if file doesn't exist or load fails
- Model is used in the LLM categorization stage of the cascading workflow

### LLM Candidate Shortlists

To reduce prompt size and token spend, the workflow feeds DSPy only the top semantic candidates per taxonomy page. Configure the shortlist via:

- `SEMANTIC_CANDIDATE_LIMIT` – maximum neighbors retrieved from Supabase (default `25`).
- `LLM_CANDIDATE_LIMIT` – maximum neighbors forwarded to the LLM fallback (default `10`).
- `LLM_CANDIDATE_MIN_SCORE` – minimum cosine similarity for inclusion (default `0.6`).

When no candidate meets the minimum score, the engine automatically backfills with the best available semantic matches so DSPy still receives context.

## Troubleshooting

**Error: "No valid examples found in dataset file"**
- Ensure dataset file contains at least one valid example
- Check that required columns/fields are present
- Verify data types (best_match_index must be integer)

**Error: "For GEPA optimizer, provide exactly one of: --budget, --max-full-evals, or --max-metric-calls"**
- GEPA requires exactly one budget specification
- Remove conflicting budget options

**Warning: "Very few training examples"**
- Increase dataset size to at least 50 examples
- Results may not be optimal with <10 examples

**Optimization fails silently**
- Check logs for detailed error messages
- Verify LLM API credentials and model availability
- Ensure reflection model is accessible (for GEPA)
