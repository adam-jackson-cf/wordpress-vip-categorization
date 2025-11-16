"""Promote a versioned matcher to matcher_latest.json with meta manifest."""

from __future__ import annotations

import json
import logging
import sys

# Add src to path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.optimization.dspy_optimizer import ACTIVE_MODEL_PATH, MODELS_DIR, DSPyOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    db = SupabaseClient(settings)
    optimizer = DSPyOptimizer(settings, db)

    # Parse CLI args (very lightweight to avoid click dependency here)
    import argparse

    parser = argparse.ArgumentParser(description="Promote optimized matcher to latest.")
    parser.add_argument("--version", type=int, help="Specific matcher version N to promote (matcher_vN.json)")
    parser.add_argument("--validation-score", type=float, help="Optional validation score to record")
    parser.add_argument("--dataset-path", type=str, help="Optional dataset path used for evaluation")
    parser.add_argument("--budget", type=str, help="Optional GEPA budget used during optimization")
    args = parser.parse_args()

    # Select source model path
    source_path: Path | None
    if args.version is not None:
        candidate = MODELS_DIR / f"matcher_v{args.version}.json"
        if not candidate.exists():
            logger.error("Specified version does not exist: %s", candidate)
            sys.exit(1)
        source_path = candidate
    else:
        source_path = optimizer._get_latest_versioned_model_path()
        if source_path is None:
            logger.error("No matcher_vN.json files available to promote.")
            sys.exit(1)

    # Basic rubric-compatibility check: try loading the model
    try:
        optimizer.load_optimized_model(str(source_path))
    except Exception as exc:
        logger.error("Failed to load candidate model %s: %s", source_path, exc)
        sys.exit(1)

    # Promote (copy) to matcher_latest.json
    ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(source_path, ACTIVE_MODEL_PATH)

    # Write meta manifest next to matcher_latest.json
    meta = {
        "version": args.version if args.version is not None else source_path.stem.replace("matcher_v", ""),
        "source_file": str(source_path.name),
        "validation_score": args.validation_score,
        "dataset_path": args.dataset_path,
        "budget": args.budget,
    }
    meta_path = ACTIVE_MODEL_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Promoted {source_path.name} -> {ACTIVE_MODEL_PATH.name}")
    print(f"Wrote meta manifest: {meta_path.name}")


if __name__ == "__main__":
    main()
