# DSPy & Optimization Rules

Applies to modules in `src/optimization/` plus DSPy helpers referenced by CLI commands.

## Signatures & Modules
- Define every prompt via `dspy.Signature` (single responsibility, well-described fields) and compose logic with `dspy.Module`.

```python
class CategorizationSignature(dspy.Signature):
    taxonomy_description: str = dspy.InputField(desc="Description of the taxonomy entry")
    content: str = dspy.InputField(desc="Body text to categorize")
    category: str = dspy.OutputField(desc="Chosen taxonomy category")
    confidence: float = dspy.OutputField(desc="0-1 confidence score")
```

## Optimization Workflow
- Prefer GEPA (or bootstrap → GEPA) for tuning; manual prompt tweaks belong only in experiments.
- Always supply an explicit metric callable and a held-out evaluation dataset; document seeds for reproducibility.

```python
optimizer = GEPA(
    metric=categorization_metric,
    breadth=10,
    depth=3,
    init_temperature=1.0,
)
optimized = optimizer.compile(student=categorizer, trainset=train_examples, eval_kwargs={"num_threads": 4})
```

## Dataset & Artifact Handling
- Training datasets live under `data/` and must represent the production taxonomy/content mix.
- Optimized artifacts version as `matcher_vN.*` and ship through `prompt-optimiser/models/`; never overwrite earlier versions.

## Checklist (`src/optimization/`)
- Every prompt is modeled with `dspy.Signature` + `dspy.Module` composition.
- Optimization scripts wire metrics, seeds, and budgets explicitly; no implicit globals.
- Optimization outputs are versioned and stored under `prompt-optimiser/` with matching config/report entries.
