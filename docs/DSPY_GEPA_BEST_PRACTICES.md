# DSPy and GEPA Best Practices for Agent Prompt Optimization

## Overview

DSPy transforms prompt engineering into systematic program optimization. GEPA (Genetic Evolutionary Prompt Optimization) uses language model-driven reflection to iteratively refine prompts. Use these patterns when implementing or optimising agent prompts.

## DSPy Fundamentals

### Signatures

Define input/output contracts using `dspy.Signature`. Use descriptive field names and clear descriptions.

```python
class TaxonomyMatcher(dspy.Signature):
    """Match a taxonomy page to the most relevant content page."""

    taxonomy_category: str = dspy.InputField(desc="Category of the taxonomy page")
    taxonomy_description: str = dspy.InputField(desc="Description of the taxonomy page")
    taxonomy_keywords: str = dspy.InputField(desc="Keywords for the taxonomy page (comma-separated)")
    content_summaries: str = dspy.InputField(desc="List of available content pages with index, title, URL, and preview")

    best_match_index: int = dspy.OutputField(desc="Index of the best matching content page, or -1 if no good match")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1) for the match")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why this match was chosen")
```

### Modules

Create reusable modules that encapsulate prediction logic. Use `dspy.ChainOfThought` for complex reasoning tasks.

```python
class MatchingModule(dspy.Module):
    """DSPy module for taxonomy-to-content matching."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(TaxonomyMatcher)

    def forward(
        self,
        taxonomy_category: str,
        taxonomy_description: str,
        taxonomy_keywords: str,
        content_summaries: str,
    ) -> dspy.Prediction:
        return self.predict(
            taxonomy_category=taxonomy_category,
            taxonomy_description=taxonomy_description,
            taxonomy_keywords=taxonomy_keywords,
            content_summaries=content_summaries,
        )
```

### Configuration

Configure DSPy with your language model provider. Support OpenAI-compatible APIs by detecting provider from base URL.

```python
import dspy

# Determine provider from base URL
model_name = settings.llm_model
if "openrouter" in settings.llm_base_url.lower():
    if not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"
elif not model_name.startswith("openai/"):
    model_name = f"openai/{model_name}"

lm = dspy.LM(
    model=model_name,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=0.3,
)
dspy.configure(lm=lm)
```

**Note**: Both `import dspy` with `dspy.LM()` and `from dspy.clients import LM` are valid; the former is more common in recent examples.

### Advanced LM Features

#### Caching and Diversity

The LM client caches responses by default to reduce API calls and costs. Control caching behaviour:

```python
# Disable caching entirely
lm = dspy.LM(model="openai/gpt-4", cache=False)

# Keep caching enabled but force new requests for diversity
# Pass unique rollout_id with non-zero temperature
prediction = model(rollout_id="attempt_2", temperature=0.7)
```

#### Automatic Retries

The LM client automatically retries failed requests due to temporary issues (network glitches, rate limits). Configure retry behaviour via `num_retries` parameter during initialisation.

#### Response Types

Enable structured response handling with the Responses API:

```python
lm = dspy.LM(model="openai/gpt-4", model_type="responses")
```

## Training Data Preparation

Convert domain data into `dspy.Example` objects. Mark input fields explicitly using `with_inputs()`.

```python
example = dspy.Example(
    taxonomy_category=taxonomy.category,
    taxonomy_description=taxonomy.description,
    taxonomy_keywords=keywords_str,
    content_summaries=content_summaries,
    best_match_index=best_match_index,
    confidence=matching.similarity_score,
    reasoning="",
).with_inputs(
    "taxonomy_category",
    "taxonomy_description",
    "taxonomy_keywords",
    "content_summaries",
)
```

## Automatic Few-Shot Learning

**Terminology Note**: DSPy is transitioning from "teleprompters" to "optimisers" terminology. Both `dspy.teleprompt` and future `dspy.optimizers` namespaces are valid; documentation and code examples may use either term interchangeably.

### BootstrapFewShot

Use `BootstrapFewShot` to automatically generate and select high-quality demonstrations from training data.

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=self.accuracy_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)

optimized = optimizer.compile(
    self.matcher,
    trainset=train_set,
)
```

### LabeledFewShot

Use `LabeledFewShot` when you have pre-labelled examples and want to select the most effective demonstrations.

```python
from dspy.teleprompt import LabeledFewShot

optimizer = LabeledFewShot(
    metric=self.accuracy_metric,
    max_labeled_demos=8,
)
```

## Evaluation Metrics

Define metrics that return float scores (0-1). Include confidence calibration to penalise overconfident wrong predictions.

```python
def accuracy_metric(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Compute accuracy metric for matching evaluation."""
    try:
        expected_index = int(example.best_match_index)
        predicted_index = int(prediction.best_match_index) if hasattr(prediction, "best_match_index") else -1
    except (ValueError, TypeError, AttributeError):
        return 0.0

    index_match = 1.0 if predicted_index == expected_index else 0.0

    # Confidence calibration: penalise overconfidence on wrong matches
    confidence_penalty = 0.0
    if hasattr(prediction, "confidence"):
        try:
            conf = float(prediction.confidence)
            if index_match == 0.0 and conf > 0.8:
                confidence_penalty = 0.2
        except (ValueError, TypeError):
            pass

    return max(0.0, index_match - confidence_penalty)
```

Use `Evaluate` to validate optimised models on held-out data.

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=val_set,
    metric=self.accuracy_metric,
    num_threads=1,
    display_progress=True,
    display_table=5,  # Show first 5 examples with scores (0 = no table)
)

score = evaluator(optimized)
```

**Evaluation Parameters**:
- `num_threads`: Parallel evaluation threads (start with 1 for debugging, scale to 4-24 for production)
- `display_progress`: Show progress bar during evaluation
- `display_table`: Number of example results to display (0 disables detailed output)

## GEPA Configuration

GEPA uses genetic algorithms with language model reflection to evolve better prompts. Configure budget, reflection, and evaluation settings.

### Performance Benchmarks

GEPA (Agrawal et al., 2025) demonstrates significant improvements over existing optimisation methods:
- **10-20% improvement** over GRPO (Gradient-based Policy Optimisation) across four tasks
- **35x fewer rollouts** required compared to GRPO for equivalent performance
- **10%+ improvement** over MIPROv2 (leading prompt optimiser) across two LLMs
- **AIME 2025 benchmark**: Improved GPT-4.1 Mini from 46.6% to 56.6% accuracy

GEPA maintains a **Pareto frontier** - the set of candidates achieving the highest score on at least one evaluation instance. Each iteration samples the next mutation candidate from this frontier, ensuring diverse exploration whilst retaining high-performing variants.

### Budget Configuration

Choose budget based on optimisation needs:
- `Light`: Quick experimentation (approximately 500-1600 metric calls, 6-12 full evaluations)
- `Medium`: Balanced optimisation (moderate metric calls and generations)
- `Heavy`: Thorough optimisation (more generations, higher cost, comprehensive search)

Exactly one of `auto`, `max_full_evals`, or `max_metric_calls` must be provided. Use `auto="light"` for quick experiments; increase to `"medium"` or `"heavy"` for production optimisation.

### Reflection Configuration

Use a robust language model for reflection. Higher temperature (1.0) encourages diverse prompt proposals.

```python
from dspy.teleprompt import GEPA

reflection_lm = dspy.LM(
    model='gpt-4',
    temperature=1.0,
    max_tokens=32000
)

optimizer = GEPA(
    reflection_lm=reflection_lm,
    budget="medium",
    use_merge=True,  # Combine successful program variants
)
```

### Evaluation Configuration

Configure parallel evaluation and metric ranges for efficient optimisation.

```python
optimizer = GEPA(
    metric=self.accuracy_metric,
    num_threads=4,  # Parallel evaluation
    failure_score=0.0,  # Minimum acceptable score
    perfect_score=1.0,  # Target score
    log_dir="./gepa_logs",
    track_stats=True,  # Comprehensive logging
    seed=42,  # Reproducibility
)
```

### Custom Instruction Proposers

Customise instruction proposers to analyse execution traces and generate improved prompts. GEPA accepts text feedback to guide optimisation, providing visibility into why the system achieved specific scores.

```python
def custom_instruction_proposer(traces, feedback):
    """Analyse execution traces and propose improved instructions."""
    # Analyse patterns in successful vs failed traces
    # Generate instruction improvements using reflection_lm
    return improved_instructions

optimizer = GEPA(
    instruction_proposer=custom_instruction_proposer,
    reflection_lm=reflection_lm,
)
```

GEPA's reflective approach incorporates natural language feedback to learn high-level rules from trial and error, making it more interpretable than gradient-based methods relying on sparse scalar rewards.

## Integration Patterns

### Model Persistence

Save and load optimised models for reuse. DSPy supports both JSON and pickle formats.

```python
# Save to JSON (default, human-readable) in prompt-optimiser directory
model.save("prompt-optimiser/models/matcher_vN.json")

# Save to pickle (for non-JSON-serializable objects)
model.save("prompt-optimiser/models/matcher_vN.pkl")

# Save entire program including structure
model.save("prompt-optimiser/models/matcher_vN.json", save_program=True)

# Load (automatically detects format from extension)
self.matcher.load("prompt-optimiser/models/matcher_vN.json")
```

**Format Selection**:
- **JSON**: Plain-text format, human-readable, includes optimised prompts and demonstrations
- **Pickle**: Binary format, handles complex objects that can't be JSON-serialised
- If you encounter serialisation warnings with JSON, switch to `.pkl` or use `save_program=True`

**What Gets Saved**:
- BootstrapFewShot: All bootstrapped examples added to prompts
- COPRO/MIPROv2/GEPA: Optimised instructions and prompt improvements
- Model state preserves optimisation work, enabling reuse without re-running expensive optimisation

### Validation Set Splitting

Split training data when validation set is not provided.

```python
if validation_examples is None:
    split_point = int(len(training_examples) * 0.8)
    train_set = training_examples[:split_point]
    val_set = training_examples[split_point:]
else:
    train_set = training_examples
    val_set = validation_examples
```

### Error Handling

Handle optimisation failures gracefully, returning unoptimised model as fallback.

```python
try:
    optimized = optimizer.compile(self.matcher, trainset=train_set)
    return optimized
except Exception as e:
    logger.error(f"Optimisation failed: {e}")
    return self.matcher  # Return unoptimised fallback
```

## Multi-Metric Optimization

Combine multiple evaluation dimensions when optimising complex tasks.

```python
def composite_metric(self, example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Combine accuracy, confidence calibration, and reasoning quality."""
    accuracy_score = self.accuracy_metric(example, prediction)
    confidence_score = self.confidence_metric(example, prediction)
    reasoning_score = self.reasoning_quality_metric(example, prediction)

    # Weighted combination
    return 0.6 * accuracy_score + 0.2 * confidence_score + 0.2 * reasoning_score
```

## DSPy Ecosystem (2025)

### Version Context

Current stable versions: DSPy 2.5/2.6 (as of early 2025), with version 3.0 approaching. The framework has matured significantly, with comprehensive integration support and production-ready optimisation capabilities.

### Additional Optimisers

Beyond BootstrapFewShot and GEPA, the 2025 ecosystem includes:
- **MIPROv2**: Leading instruction optimiser using multi-step prompt refinement
- **OPRO**: Optimisation through prompting
- **BootstrapFinetune**: Combines few-shot learning with model fine-tuning
- **BetterTogether**: Ensemble optimisation strategies

### Structured Outputs and Type Safety

DSPy integrates well with **Pydantic** for type-safe structured outputs. Combine DSPy signatures with Pydantic models for robust data validation and type checking in production applications.

```python
from pydantic import BaseModel

class MatchResult(BaseModel):
    best_match_index: int
    confidence: float
    reasoning: str

# Use with DSPy for type-safe outputs
```

### Framework Positioning

DSPy positions itself as an **optimiser framework** rather than an orchestration tool. Whilst frameworks like LangChain and LlamaIndex focus on connecting components and managing workflows, DSPy specialises in systematically improving AI system quality through optimised prompts and fine-tuned weights.

## Best Practices Summary

1. **Use descriptive signatures**: Clear input/output field descriptions guide the model
2. **Leverage ChainOfThought**: Enable step-by-step reasoning for complex tasks
3. **Prepare quality training data**: Filter successful matches, format consistently
4. **Configure appropriate optimisers**: `BootstrapFewShot` for automatic demonstrations, GEPA for reflective evolution
5. **Define calibrated metrics**: Include confidence penalties to prevent overconfidence
6. **Validate on held-out data**: Always evaluate optimised models before deployment
7. **Enable logging and tracking**: Use `track_stats=True` and `log_dir` for debugging
8. **Set seeds for reproducibility**: Ensure consistent optimisation results
9. **Handle failures gracefully**: Return unoptimised fallback on errors
10. **Support multiple providers**: Detect provider from base URL for flexibility
