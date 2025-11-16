"""Promote the latest versioned matcher to matcher_latest.json."""

from __future__ import annotations

import logging
import sys

# Add src to path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.optimization.dspy_optimizer import ACTIVE_MODEL_PATH, DSPyOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    db = SupabaseClient(settings)
    optimizer = DSPyOptimizer(settings, db)

    promoted_path = optimizer.promote_latest_model()
    if promoted_path is None:
        logger.error("No matcher_vN.json files available to promote.")
        sys.exit(1)

    print(f"Promoted optimized matcher to {ACTIVE_MODEL_PATH}")


if __name__ == "__main__":
    main()
