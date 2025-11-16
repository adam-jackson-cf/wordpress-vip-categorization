# DSPy Optimization Quick Start Guide

## Low-Cost Quick Run Command

For a low-cost, quick optimization run to understand timings and costs, use the **`bootstrap`** optimizer with minimal demonstrations:

```bash
python -m src.cli optimize-dataset \
  --dataset data/dspy_training_dataset.csv \
  --optimizer bootstrap \
  --train-split 0.2 \
  --report prompt-optimiser/reports/report_vN.md \
  --seed 42
```

### Command Breakdown

- **`--dataset data/dspy_training_dataset.csv`**: Uses the curated 360-example dataset we generated
- **`--optimizer bootstrap`**: Uses `BootstrapFewShot` (fastest, lowest cost option)
  - **Cost**: ~20-40 LLM calls (much cheaper than GEPA)
  - **Time**: ~2-5 minutes for 100 examples
  - **Best for**: Quick experimentation and understanding the process
- **`--train-split 0.2`**: Uses 20 examples for training, 80 for validation
- **`--output`**: Optional path; if omitted, the model is saved to `prompt-optimiser/models/matcher_vN.json`
- **`--report prompt-optimiser/reports/report_vN.md`**: Example of a versioned markdown report path (you choose `vN`)
- **`--seed 42`**: Ensures reproducible results

## Alternative: Even Faster (Minimal Demos)

For the absolute fastest run with minimal cost:

```bash
python -m src.cli optimize-dataset \
  --dataset data/dspy_training_dataset.csv \
  --optimizer bootstrap \
  --train-split 0.2 \
  --report prompt-optimiser/reports/report_vN.md \
  --seed 42
```

Note: The bootstrap optimizer uses default `max_bootstrapped_demos=4` and `max_labeled_demos=8`, which is already quite minimal.

## What the Report Contains

The `optimization_report.md` file will include:

1. **Summary**: Optimizer type, dataset sizes, validation score, duration
2. **Configuration**: All optimizer settings and model configuration
3. **Before Optimization**: Initial prompt instructions and demonstration count
4. **After Optimization**: Optimized prompt instructions and demonstration count
5. **Changes**: Summary of what changed (instructions, demonstrations, etc.)
6. **Notes**: Additional context about the optimization process

## Cost Comparison

| Optimizer | Estimated LLM Calls | Estimated Time | Use Case |
|-----------|-------------------|----------------|----------|
| `bootstrap` | 20-40 | 2-5 min | Quick experimentation |
| `bootstrap-random-search` | 50-100 | 5-10 min | Better results, still fast |
| `gepa` (light) | 500-1600 | 15-30 min | Thorough optimization |
| `gepa` (medium) | 2000-5000 | 30-60 min | Production quality |
| `gepa` (heavy) | 5000+ | 60+ min | Maximum quality |

## Smoke Test Script (Bootstrap)

When you modify the categorization workflow or DSPy integration, run the fast smoke test to make sure nothing fundamental broke before spending money on GEPA:

```bash
python scripts/run_quick_optimization_test.py \
  --dataset data/dspy_training_dataset.csv \
  --train-split 0.2 \
  --seed 42 \
  --max-examples 30
```

This script is the old “Stage 1” bootstrap step pulled out of the workflow:

- Uses DSPy BootstrapFewShot with 4 bootstrapped / 8 labeled demos.
- Finishes in seconds and costs ~30 LLM calls per 100 examples.
- Stores artifacts as `_test` files (`matcher_test.json`, `dspy_config_test.json`, `report_test.md`), overwriting previous smoke tests.
- Intended purely as a sanity check—scores from this run are not production-quality, but they prove the stack works end-to-end on localhost.

## GEPA Optimization Workflow

For the real optimization run (previously “Stage 2”), use:

```bash
python scripts/run_optimization_workflow.py \
  --dataset data/dspy_training_dataset.csv \
  --train-split 0.2 \
  --seed 42 \
  --budget medium
```

### Workflow Behavior

- Runs the GEPA optimizer with the requested budget (`light`, `medium`, or `heavy`).
- Writes versioned artifacts (`matcher_vN.json`, `dspy_config_vN.json`, `report_vN.md`).
- Prints validation score, runtime, and a rough metric-call estimate at the end.
- Because there is no staging environment, versioned artifacts stay in `prompt-optimiser/` until you explicitly promote one (see below); the categorization service reads `matcher_latest.json`.

### Options

- `--dataset`: CSV/JSON training dataset (required).
- `--train-split`: Train/validation split (default `0.2`).
- `--seed`: Random seed (default `42`).
- `--budget`: `light`, `medium`, or `heavy` preset (default `medium`).
- `--max-examples`: Optional slice for experimentation.
- `--num-threads`: Parallel GEPA evaluation threads if you have local capacity.

### Local “Promotion” Checklist

1. Run `scripts/run_quick_optimization_test.py` after code changes to validate the bootstrap flow.
2. Run `scripts/run_optimization_workflow.py` with the same dataset to produce a GEPA candidate (versioned `matcher_vN.json`).
3. When you decide a version is ready for production, run `scripts/promote_optimized_model.py` to copy the latest `matcher_vN.json` to `matcher_latest.json`.
4. Rerun the end-to-end CLI locally (e.g., `python -m src.cli workflow start ...`) to confirm the promoted prompt behaves as expected before committing.

If you prefer to manage artifacts manually, you can still use `python -m src.cli optimize-dataset ...` with explicit budgets—both scripts ultimately call the same optimizer under the hood. Promotion simply controls which artifact `CategorizationService` loads.

### Running Heavy Budgets Reliably

- **Clear GEPA state first:** `rm -f prompt-optimiser/gepa_logs/gepa_state.bin` before every new heavy run so you don’t resume an old search.
- **Use tmux for long runs:** Heavy budgets easily exceed a couple of minutes, so wrap the command in a tmux session, e.g.

  ```bash
  tmux new-session -d -s gepa_heavy \
    'cd /Users/you/path && rm -f prompt-optimiser/gepa_logs/gepa_state.bin && \
     python scripts/run_optimization_workflow.py --dataset data/dspy_training_dataset.csv \
       --train-split 0.2 --seed 42 --budget heavy --num-threads 4'
  tmux attach -t gepa_heavy  # watch progress / scroll back after completion
  ```

- Kill the session (or let it exit naturally) once the run finishes, then check `prompt-optimiser/models/` and `reports/` for the new version.

## Next Steps

After running the quick optimization:

1. **Review the report**: Check `optimization_report.md` to see what changed
2. **Test the model**: The optimized model is saved to `prompt-optimiser/models/matcher_vN.json` (version number auto-incremented)
3. **Compare results**: Run validation to see improvement in accuracy
4. **Scale up**: If results look good, run with `gepa` optimizer for better quality

## Example Output

After running, you'll see:

```
Loading dataset from data/dspy_training_dataset.csv...
Optimizing with 360 examples using bootstrap optimizer...
⚠ This process is expensive by nature (multiple iterations, metric evaluations, LLM calls).
✓ Optimization complete. Model saved to prompt-optimiser/models/matcher_vN.json
✓ Optimization report saved to prompt-optimiser/reports/report_vN.md
```

The report will show before/after prompts, configuration changes, and performance metrics.
