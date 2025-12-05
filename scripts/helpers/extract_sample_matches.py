#!/usr/bin/env python3
"""Extract random sample of semantic matches for analysis."""

import argparse
import csv
import random
from pathlib import Path


def extract_sample_matches(
    input_file: str,
    output_file: str,
    min_score: float = 0.5,
    max_score: float = 0.75,
    sample_size: int = 50,
) -> None:
    """Extract random sample of semantic matches within score range.

    Args:
        input_file: Path to match snapshot CSV
        output_file: Path to output sample CSV
        min_score: Minimum similarity score (inclusive)
        max_score: Maximum similarity score (inclusive)
        sample_size: Number of rows to sample
    """
    # Read and filter matching rows
    filtered_rows = []

    with open(input_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["match_stage"] == "semantic_matched":
                try:
                    score = float(row["similarity_score"])
                    if min_score <= score <= max_score:
                        filtered_rows.append(row)
                except (ValueError, KeyError):
                    continue

    print(f"Found {len(filtered_rows)} rows matching criteria")

    # Random sample
    if len(filtered_rows) > sample_size:
        sampled_rows = random.sample(filtered_rows, sample_size)
    else:
        sampled_rows = filtered_rows
        print(f"Warning: Only {len(filtered_rows)} rows available (requested {sample_size})")

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        if sampled_rows:
            writer = csv.DictWriter(f, fieldnames=sampled_rows[0].keys())
            writer.writeheader()
            writer.writerows(sampled_rows)

    print(f"Wrote {len(sampled_rows)} rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random sample of semantic matches")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to match snapshot CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/examples/sample_matches.csv",
        help="Path to output sample CSV (default: data/examples/sample_matches.csv)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum similarity score (default: 0.5)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.75,
        help="Maximum similarity score (default: 0.75)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows to sample (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    extract_sample_matches(
        args.input,
        args.output,
        min_score=args.min_score,
        max_score=args.max_score,
        sample_size=args.sample_size,
    )
