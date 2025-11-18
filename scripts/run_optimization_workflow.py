"""GEPA optimization workflow.

Runs the thorough GEPA optimizer against the provided dataset and saves
versioned artifacts. Use scripts/run_quick_optimization_test.py for the fast
bootstrap smoke test before invoking this workflow.
"""

import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal

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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def estimate_gepa_cost(num_examples: int, budget: Literal["light", "medium", "heavy"]) -> str:
    """Rough cost estimate for GEPA budgets."""
    # Heuristic call counts pulled from docs/OPTIMIZATION_QUICKSTART.md
    per_100 = {"light": 1200, "medium": 3500, "heavy": 6000}
    estimated_calls = int((num_examples / 100) * per_100[budget])
    return f"~{estimated_calls} metric calls"


def run_gepa(
    optimizer: DSPyOptimizer,
    training_data: list,
    train_split: float,
    seed: int,
    budget: Literal["light", "medium", "heavy"],
    num_threads: int | None,
) -> tuple[float, float | None, Path, Path, Path]:
    """Run GEPA optimization with the requested budget."""
    version = optimizer._get_next_version("unused")
    model_path = MODELS_DIR / f"matcher_v{version}.json"
    report_path = REPORTS_DIR / f"report_v{version}.md"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_size = int(len(training_data) * train_split)
    val_size = len(training_data) - train_size

    logger.info("=" * 70)
    logger.info("GEPA OPTIMIZATION")
    logger.info("=" * 70)
    logger.info("Training examples: %s", train_size)
    logger.info("Validation examples: %s", val_size)
    logger.info("Budget: %s", budget)
    logger.info("Seed: %s", seed)

    optimizer_kwargs: dict[str, int | str] = {"budget": budget}
    if num_threads is not None:
        optimizer_kwargs["num_threads"] = num_threads

    before_model = deepcopy(optimizer.matcher)
    start_time = time.time()

    optimized_model = optimizer.optimize_with_dataset(
        training_data,
        optimizer_type="gepa",
        **optimizer_kwargs,  # type: ignore[arg-type]
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
    logger.info("✓ Model saved to %s", model_path)

    optimizer_config = {
        "optimizer": "gepa",
        "budget": budget,
        "train_split": train_split,
        "num_threads": num_threads or optimizer.settings.dspy_num_threads,
        "seed": seed,
    }

    try:
        config_file = optimizer.save_optimization_config(
            before_model=before_model,
            after_model=optimized_model,
            optimizer_type="gepa",
            optimizer_config=optimizer_config,
            training_size=train_size,
            validation_size=val_size,
            model_file_path=str(model_path),
            validation_score=float(validation_score) if validation_score is not None else None,
            duration_seconds=duration,
        )
        config_path = CONFIGS_DIR / f"dspy_config_v{version}.json"
        if config_file != config_path:
            config_file.rename(config_path)
        logger.info("✓ Config saved to %s", config_path)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.warning("Could not save optimization config: %s", exc)
        config_path = CONFIGS_DIR / f"dspy_config_v{version}.json"

    report_content = optimizer.generate_optimization_report(
        before_model=before_model,
        after_model=optimized_model,
        optimizer_type="gepa",
        optimizer_config=optimizer_config,
        training_size=train_size,
        validation_size=val_size,
        validation_score=float(validation_score) if validation_score is not None else None,
        duration_seconds=duration,
    )
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("✓ Report saved to %s", report_path)

    return duration, validation_score, model_path, config_path, report_path


def main() -> None:
    """Main workflow function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Full GEPA optimization workflow (use run_quick_optimization_test.py for smoke tests)"
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
        help="Limit number of examples used for testing",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA budget preset (default: medium)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Parallel evaluation threads (default from settings)",
    )

    args = parser.parse_args()

    # Validate train split
    if not 0 < args.train_split < 1:
        logger.error("Train split must be between 0 and 1")
        sys.exit(1)

    # Validate dataset file exists
    if not args.dataset.exists():
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Load settings and initialize optimizer
    settings = get_settings()
    db = SupabaseClient(settings)
    optimizer = DSPyOptimizer(settings, db)

    # Override seed if provided
    settings.dspy_optimization_seed = args.seed
    settings.dspy_train_split_ratio = args.train_split

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    try:
        training_data = optimizer.load_training_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    # Limit examples if requested (useful for testing)
    if args.max_examples and args.max_examples < len(training_data):
        training_data = training_data[: args.max_examples]
        logger.info(f"Limited to {len(training_data)} examples for testing")

    if len(training_data) < 10:
        logger.warning("Very few training examples. Results may not be optimal.")
        logger.warning("Recommend at least 50 examples, optimal for 300+ examples.")

    logger.info(f"Using {len(training_data)} examples from dataset")

    duration, score, model_path, config_path, report_path = run_gepa(
        optimizer=optimizer,
        training_data=training_data,
        train_split=args.train_split,
        seed=args.seed,
        budget=args.budget,
        num_threads=args.num_threads,
    )

    print("\n" + "=" * 70)
    print("GEPA RESULTS SUMMARY")
    print("=" * 70)
    if score is not None:
        print(f"Validation Score: {score:.3f}")
    else:
        print("Validation Score: N/A")
    print(f"Duration: {format_duration(duration)}")
    print(f"Estimated Cost: {estimate_gepa_cost(len(training_data), args.budget)}")
    print("\nArtifacts:")
    print(f"  Model:   {model_path}")
    print(f"  Config:  {config_path}")
    print(f"  Report:  {report_path}")
    print("=" * 70)

    logger.info("\n✓ GEPA optimization workflow completed successfully!")


if __name__ == "__main__":
    main()
