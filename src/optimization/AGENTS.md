# DSPy & Prompt Optimization Standards

Project overview: [../../AGENTS.md](../../AGENTS.md)
Code review checklist: [../../CODE_REVIEW.md](../../CODE_REVIEW.md)
Quick start: [../../docs/OPTIMIZATION_QUICKSTART.md](../../docs/OPTIMIZATION_QUICKSTART.md)
Best practices: [../../docs/DSPY_GEPA_BEST_PRACTICES.md](../../docs/DSPY_GEPA_BEST_PRACTICES.md)

## DSPy Signature Design

```python
import dspy

class CategorizationSignature(dspy.Signature):
    """Categorize content against taxonomy pages."""

    taxonomy_description: str = dspy.InputField(
        desc="Description of the taxonomy category"
    )
    content: str = dspy.InputField(
        desc="Content to categorize"
    )

    category: str = dspy.OutputField(desc="Matched category name")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    reasoning: str = dspy.OutputField(desc="Explanation for the match")
```

Use `dspy.Signature` for all LLM prompts - provides structure and validation
Keep signatures focused - single responsibility per signature
Use `dspy.Module` for composable components

## GEPA Optimization Pattern

```python
from dspy.teleprompt import GEPA
from dspy.evaluate import Evaluate

def categorization_metric(example, prediction, trace=None) -> float:
    return float(example.category == prediction.category)

optimizer = GEPA(
    metric=categorization_metric,
    breadth=10,
    depth=3,
    init_temperature=1.0
)

optimized_module = optimizer.compile(
    student=categorization_module,
    trainset=train_examples,
    eval_kwargs={"num_threads": 4}
)
```

Use GEPA for prompt optimization - not manual prompt engineering
Prepare diverse training datasets with representative examples
Use appropriate metrics for evaluation (EM, F1, custom)
Keep optimization reproducible - set random seeds

## Dataset Generation

See `scripts/generate_dspy_dataset.py` for creating training datasets:

```bash
python scripts/generate_dspy_dataset.py \
    --output data/dspy_training_set.json \
    --taxonomy-file data/taxonomy.csv \
    --num-examples 50
```

Datasets should be diverse and representative of production scenarios.

## Optimization Workflow

1. **Define Signature**: Create focused `dspy.Signature` for task
2. **Create Module**: Build `dspy.Module` using signatures
3. **Generate Dataset**: Create diverse training examples
4. **Define Metric**: Write evaluation function
5. **Run GEPA**: Optimize with appropriate hyperparameters
6. **Evaluate**: Test on held-out validation set
7. **Deploy**: Use optimized module in production

## Comprehensive Guidance

For complete DSPy implementation patterns, best practices, and troubleshooting:
- **Quick Start**: [../../docs/OPTIMIZATION_QUICKSTART.md](../../docs/OPTIMIZATION_QUICKSTART.md)
- **Best Practices**: [../../docs/DSPY_GEPA_BEST_PRACTICES.md](../../docs/DSPY_GEPA_BEST_PRACTICES.md)
- **Implementation Details**: [../../docs/DSPY_IMPLEMENTATION.md](../../docs/DSPY_IMPLEMENTATION.md)

## Common Pitfalls

1. **Manual Prompt Engineering**: Use GEPA optimization instead
2. **Poor Training Data**: Ensure diverse, representative examples
3. **Wrong Metrics**: Match metric to task (EM for exact, F1 for partial)
4. **Non-Reproducible**: Always set random seeds for optimization
5. **Overfitting**: Use validation set separate from training data

## DSPy Checklist (src/optimization)

- [ ] LLM prompts are defined via `dspy.Signature` and composed in `dspy.Module`
- [ ] Each signature has a single, clear responsibility and well-described fields
- [ ] Optimization uses GEPA (or similar) with an explicit metric function
- [ ] Training datasets are diverse and representative of production use cases
- [ ] Random seeds are set for reproducible optimization runs
- [ ] Optimized modules are evaluated on held-out validation data before use in production
