#!/usr/bin/env python
"""Compute semantic coverage metrics for various similarity thresholds."""

# ruff: noqa: E402  # requires sys.path mutation before importing project modules

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchStage

THRESHOLDS = [0.60, 0.65, 0.70, 0.75]


def _candidate_score(match) -> float:
    return float(match.semantic_similarity_score or 0.0)


def main() -> None:
    settings = get_settings()
    db = SupabaseClient(settings)
    matches = db.get_all_matchings()

    if not matches:
        print("No matching_results rows found; run the matching workflow first.")
        return

    stage_counts = Counter(match.match_stage for match in matches)
    total = len(matches)
    print(f"Total current matches: {total}")
    print("Stage breakdown:")
    for stage, count in stage_counts.items():
        label = stage.value if isinstance(stage, MatchStage) else stage
        print(f"  - {label}: {count}")

    print("\nThreshold coverage (semantic_similarity >= threshold):")
    header = "Threshold  Covered  Coverage%"
    print(header)
    print("-" * len(header))

    scores = [_candidate_score(match) for match in matches]
    for threshold in THRESHOLDS:
        covered = sum(1 for score in scores if score >= threshold)
        pct = (covered / total) * 100 if total else 0.0
        print(f" {threshold:0.2f}      {covered:3d}     {pct:5.1f}%")

    mean_score = sum(scores) / total
    variance = sum((score - mean_score) ** 2 for score in scores) / max(total - 1, 1)
    std_dev = math.sqrt(variance)
    print("\nCandidate similarity stats:")
    print(f"  - Mean: {mean_score:.3f}")
    print(f"  - Std Dev: {std_dev:.3f}")


if __name__ == "__main__":
    main()
