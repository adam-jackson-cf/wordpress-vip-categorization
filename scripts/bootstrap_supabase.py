#!/usr/bin/env python3
"""Automate Supabase reinitialization + pipeline smoke tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_TESTS = [
    "tests/integration/test_full_pipeline.py::TestFullPipeline::test_ingestion_with_public_site",
    "tests/integration/test_full_pipeline.py::TestFullPipeline::test_semantic_matching_integration",
]


def run_step(description: str, command: Sequence[str], env: dict[str, str] | None = None) -> None:
    print(f"\n==> {description}")
    print(" ", " ".join(command))
    completed = subprocess.run(command, env=env, check=True)
    if completed.returncode == 0:
        print("✓", description)


def build_cli_command(*parts: str) -> list[str]:
    return [sys.executable, "-m", "src.cli", *parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Supabase with schema + smoke tests")
    parser.add_argument("--taxonomy-file", type=Path, help="Override taxonomy CSV path")
    parser.add_argument("--sites", help="Comma-separated list of WordPress sites to ingest")
    parser.add_argument("--max-pages", type=int, help="Max pages per site during ingestion")
    parser.add_argument("--since", help="ISO timestamp for ingestion cutoff")
    parser.add_argument("--resume", action="store_true", help="Resume ingestion using checkpoints")
    parser.add_argument("--no-batch", dest="batch", action="store_false", help="Disable batch matching")
    parser.set_defaults(batch=True)
    parser.add_argument("--skip-init", action="store_true", help="Skip schema init")
    parser.add_argument("--skip-taxonomy", action="store_true", help="Skip taxonomy load")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--skip-match", action="store_true", help="Skip match workflow")
    parser.add_argument("--run-tests", action="store_true", help="Run integration smoke tests after matching")
    parser.add_argument(
        "--include-slow-test",
        action="store_true",
        help="Include the slow full E2E pipeline test (requires RUN_SLOW_TESTS=1)",
    )
    args = parser.parse_args()

    if not args.skip_init:
        run_step("Apply Supabase schema", build_cli_command("init-db"))

    if not args.skip_taxonomy:
        taxonomy_cmd = ["load-taxonomy"]
        if args.taxonomy_file:
            taxonomy_cmd += ["--taxonomy-file", str(args.taxonomy_file)]
        run_step("Load taxonomy", build_cli_command(*taxonomy_cmd))

    if not args.skip_ingest:
        ingest_cmd = ["ingest"]
        if args.sites:
            ingest_cmd += ["--sites", args.sites]
        if args.max_pages is not None:
            ingest_cmd += ["--max-pages", str(args.max_pages)]
        if args.since:
            ingest_cmd += ["--since", args.since]
        if args.resume:
            ingest_cmd.append("--resume")
        run_step("Ingest WordPress content", build_cli_command(*ingest_cmd))

    if not args.skip_match:
        match_cmd = ["match"]
        if not args.batch:
            match_cmd.append("--no-batch")
        run_step("Run cascading match workflow", build_cli_command(*match_cmd))

    if args.run_tests:
        tests = list(DEFAULT_TESTS)
        if args.include_slow_test:
            tests.append(
                "tests/integration/test_full_pipeline.py::TestFullPipeline::test_full_e2e_pipeline"
            )
        env = os.environ.copy()
        if args.include_slow_test:
            env["RUN_SLOW_TESTS"] = "1"
        run_step("Run integration smoke tests", ["pytest", "-n", "0", *tests], env=env)

    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
