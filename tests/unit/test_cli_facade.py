"""Unit tests for CLI facade helpers."""

from pathlib import Path

import click
import pytest

from src.services.cli_facade import (
    OptimizeDatasetOptions,
    execute_optimize_dataset,
    resolve_site_tokens_override,
)


def test_resolve_site_tokens_override_requires_tokens() -> None:
    default = [("https://example.com", "token")]
    with pytest.raises(click.BadParameter):
        resolve_site_tokens_override("https://missing.com", default)


def test_execute_optimize_dataset_budget_conflict_raises(monkeypatch) -> None:
    options = OptimizeDatasetOptions(
        dataset=Path("dummy.csv"),
        optimizer="gepa",
        budget="light",
        max_full_evals=10,
        max_metric_calls=None,
        train_split=None,
        output=Path("out.json"),
        save_program=False,
        num_threads=None,
        display_table=None,
        seed=None,
        report=None,
        no_save_config=False,
    )

    with pytest.raises(click.UsageError):
        execute_optimize_dataset(options)


def test_execute_optimize_dataset_invalid_split_raises() -> None:
    options = OptimizeDatasetOptions(
        dataset=Path("dummy.csv"),
        optimizer="gepa",
        budget=None,
        max_full_evals=None,
        max_metric_calls=None,
        train_split=1.5,
        output=Path("out.json"),
        save_program=False,
        num_threads=None,
        display_table=None,
        seed=None,
        report=None,
        no_save_config=False,
    )

    with pytest.raises(click.BadParameter):
        execute_optimize_dataset(options)
