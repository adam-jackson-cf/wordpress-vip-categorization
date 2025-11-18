"""Quick bootstrap optimization smoke test.

Runs the lightweight DSPy bootstrap optimizer against a dataset to confirm
that prompt generation, evaluation, and artifact saving all work before
running the longer GEPA workflow. Artifacts are overwritten using *_test names.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.optimization.dspy_optimizer import (
    CONFIGS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    DSPyOptimizer,
)

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    hours = seconds / 3600
    return f"{hours:.1f} hours"


def run_bootstrap_test(
    optimizer: DSPyOptimizer,
    training_data: list,
    train_split: float,
    seed: int,
) -> tuple[float, float | None, Path, Path, Path]:
    """Execute the bootstrap optimizer and persist artifacts with *_test names."""
    model_path = MODELS_DIR / "matcher_test.json"
    report_path = REPORTS_DIR / "report_test.md"
    config_path = CONFIGS_DIR / "dspy_config_test.json"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_size = int(len(training_data) * train_split)
    val_size = len(training_data) - train_size

    before_model = deepcopy(optimizer.matcher)
    start_time = time.time()

    optimized_model = optimizer.optimize_with_dataset(
        training_data,
        optimizer_type="bootstrap",
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
    )

    duration = time.time() - start_time
    validation_score: float | None = None
    try:
        from dspy.evaluate import Evaluate

        split_point = int(len(training_data) * train_split)
        val_set = training_data[split_point:]
        evaluator = Evaluate(
            devset=val_set,
            metric=optimizer.metric_fn,
            num_threads=1,
            display_progress=False,
        )
        validation_score = evaluator(optimized_model)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.warning("Could not compute validation score: %s", exc)

    optimizer.save_optimized_model(optimized_model, str(model_path))

    optimizer_config = {
        "optimizer": "bootstrap",
        "budget": None,
        "max_full_evals": None,
        "max_metric_calls": None,
        "train_split": train_split,
        "num_threads": 1,
        "seed": seed,
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 8,
    }

    config_payload = optimizer.extract_optimization_config(
        before_model=before_model,
        after_model=optimized_model,
        optimizer_type="bootstrap",
        optimizer_config=optimizer_config,
        training_size=train_size,
        validation_size=val_size,
        model_file_path=str(model_path),
        validation_score=float(validation_score) if validation_score is not None else None,
        duration_seconds=duration,
    )
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_payload, config_file, indent=2, ensure_ascii=False)

    report_content = optimizer.generate_optimization_report(
        before_model=before_model,
        after_model=optimized_model,
        optimizer_type="bootstrap",
        optimizer_config=optimizer_config,
        training_size=train_size,
        validation_size=val_size,
        validation_score=float(validation_score) if validation_score is not None else None,
        duration_seconds=duration,
    )
    report_path.write_text(report_content, encoding="utf-8")
    return duration, validation_score, model_path, config_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap-only optimization smoke test (fast sanity check)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to CSV or JSON training dataset file",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.2,
        help="Training set split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Limit number of examples used for the smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0 < args.train_split < 1:
        logger.error("Train split must be between 0 and 1")
        sys.exit(1)

    if not args.dataset.exists():
        logger.error("Dataset file not found: %s", args.dataset)
        sys.exit(1)

    settings = get_settings()
    db = SupabaseClient(settings)
    optimizer = DSPyOptimizer(settings, db)

    settings.dspy_optimization_seed = args.seed
    settings.dspy_train_split_ratio = args.train_split

    logger.info("Loading dataset from %s...", args.dataset)
    try:
        training_data = optimizer.load_training_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Error loading dataset: %s", exc)
        sys.exit(1)

    if args.max_examples and args.max_examples < len(training_data):
        training_data = training_data[: args.max_examples]
        logger.info("Limited to %s examples for testing", len(training_data))

    if len(training_data) < 10:
        logger.warning("Very few training examples. Results may not be optimal.")
        logger.warning("Recommend at least 50 examples, optimal for 300+ examples.")

    logger.info("Running bootstrap smoke test with %s examples...", len(training_data))
    duration, score, model_path, config_path, report_path = run_bootstrap_test(
        optimizer,
        training_data,
        args.train_split,
        args.seed,
    )

    print("\n" + "=" * 70)
    print("BOOTSTRAP SMOKE TEST SUMMARY")
    print("=" * 70)
    if score is not None:
        try:
            score_str = f"{float(score):.3f}"
        except (ValueError, TypeError):
            score_str = "N/A"
    else:
        score_str = "N/A"
    print(f"Validation Score: {score_str}")
    print(f"Duration: {format_duration(duration)}")
    print("Estimated Cost: ~30 LLM calls per 100 examples")
    print("\nArtifacts:")
    print(f"  Model:   {model_path}")
    print(f"  Config:  {config_path}")
    print(f"  Report:  {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
