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

- **`--dataset data/dspy_training_dataset.csv`**: Uses the 100-example dataset we generated
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
Optimizing with 100 examples using bootstrap optimizer...
⚠ This process is expensive by nature (multiple iterations, metric evaluations, LLM calls).
✓ Optimization complete. Model saved to prompt-optimiser/models/matcher_vN.json
✓ Optimization report saved to prompt-optimiser/reports/report_vN.md
```

The report will show before/after prompts, configuration changes, and performance metrics.
