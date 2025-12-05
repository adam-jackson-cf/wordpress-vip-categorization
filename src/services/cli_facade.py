"""Helper dataclasses plus orchestration helpers for complex CLI commands."""

from __future__ import annotations

import csv
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

import click

from src.config import Settings, get_settings
from src.data.supabase_client import SupabaseClient
from src.exporters.csv_exporter import CSVExporter
from src.models import MatchStage, TaxonomyPage, WordPressContent
from src.optimization.dspy_optimizer import MODELS_DIR, DSPyOptimizer
from src.services.ingestion import IngestionService
from src.services.matching import MatchingService
from src.services.workflow import WorkflowService


@dataclass(slots=True)
class MatchCommandOptions:
    threshold: float | None = None
    batch: bool = True
    skip_semantic: bool = False
    skip_llm: bool = False
    limit: int | None = None
    taxonomy_ids: str | None = None
    taxonomy_file: Path | None = None
    only_unmatched: bool = False
    force_semantic: bool = False
    force_llm: bool = False


@dataclass(slots=True)
class FullRunOptions:
    taxonomy_file: Path | None
    sites: str | None
    max_pages: int | None
    since: datetime | None
    resume: bool
    threshold: float | None
    batch: bool
    skip_semantic: bool
    skip_llm: bool
    output: Path
    min_similarity: float | None


@dataclass(slots=True)
class OptimizeDatasetOptions:
    dataset: Path
    optimizer: str
    budget: str | None
    max_full_evals: int | None
    max_metric_calls: int | None
    train_split: float | None
    output: Path
    save_program: bool
    num_threads: int | None
    display_table: int | None
    seed: int | None
    report: Path | None
    no_save_config: bool


def load_taxonomy_urls(csv_path: Path) -> list[str]:
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Destination_URL" not in reader.fieldnames:
            raise click.ClickException("Expected 'Destination_URL' column in taxonomy file")
        urls = [row.get("Destination_URL", "").strip() for row in reader]
    return [url for url in urls if url]


def resolve_site_tokens_override(
    sites_option: str | None, default_pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    if not sites_option:
        return default_pairs

    resolved: list[tuple[str, str]] = []
    token_map = dict(default_pairs)

    for raw in sites_option.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "|" in entry:
            site, token = entry.split("|", 1)
            site = site.strip().rstrip("/")
            token = token.strip()
            if not site or not token:
                raise click.BadParameter(
                    f"Invalid site entry '{raw}'. Expected format 'https://site|token'."
                )
        else:
            site = entry.rstrip("/")
            token = token_map.get(site, "")
            if not token:
                raise click.BadParameter(
                    f"No token found for '{site}'. Provide it as 'site|token' or configure it in WORDPRESS_VIP_SITE_TOKENS."
                )
        resolved.append((site, token))

    if not resolved:
        raise click.BadParameter("No valid site entries provided.")

    return resolved


def _parse_uuid_list(raw: str) -> list[UUID]:
    values: list[UUID] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(UUID(token))
        except ValueError as exc:
            raise click.BadParameter("Invalid UUID in --taxonomy-ids") from exc
    return values


def _describe_config(settings: Settings) -> None:
    click.echo(
        f"- Semantic matching: {'enabled' if settings.enable_semantic_matching else 'disabled'} "
        f"(threshold: {settings.similarity_threshold})"
    )
    click.echo(
        f"- LLM categorization: {'enabled' if settings.enable_llm_categorization else 'disabled'} (rubric-gated)"
    )


def execute_match(options: MatchCommandOptions) -> None:
    filters_selected = sum(
        bool(flag) for flag in [options.taxonomy_ids, options.taxonomy_file, options.only_unmatched]
    )
    if filters_selected > 1:
        raise click.UsageError("Use at most one of --taxonomy-ids/--taxonomy-file/--only-unmatched")

    overrides: dict[str, bool | float] = {}
    if options.skip_semantic:
        overrides["enable_semantic_matching"] = False
    if options.skip_llm:
        overrides["enable_llm_categorization"] = False
    if options.threshold is not None:
        overrides["similarity_threshold"] = options.threshold

    settings = get_settings(overrides=overrides or None)
    db = SupabaseClient(settings)
    matching_service = MatchingService(settings, db)
    workflow_service = WorkflowService(settings, db, matching_service=matching_service)

    content_subset: list[WordPressContent] | None = None
    if options.limit is not None:
        click.echo(f"Loading and limiting to first {options.limit} content items...")
        all_content = db.get_all_content()
        content_subset = all_content[: options.limit]
        click.echo(f"Selected {len(content_subset)} content items")

    taxonomy_subset: list[TaxonomyPage] | None = None
    target_ids: list[UUID] | None = None

    if options.taxonomy_ids:
        parsed_ids = _parse_uuid_list(options.taxonomy_ids)
        taxonomy_subset = db.get_taxonomy_by_ids(parsed_ids)
        target_ids = [tax.id for tax in taxonomy_subset]
        if not taxonomy_subset:
            click.echo("No taxonomy rows matched the provided IDs; nothing to do.")
            return

    elif options.taxonomy_file:
        urls = load_taxonomy_urls(options.taxonomy_file)
        if not urls:
            click.echo("No URLs found in taxonomy file; nothing to do.")
            return
        taxonomy_subset = db.get_taxonomy_by_urls(urls)
        target_ids = [tax.id for tax in taxonomy_subset]
        if not taxonomy_subset:
            click.echo("None of the provided URLs exist in the database; aborting.")
            return
        missing_count = len(set(urls) - {str(tax.destination_url) for tax in taxonomy_subset})
        if missing_count:
            click.echo(f"⚠ Skipping {missing_count} URL(s) not present in Supabase.")

    elif options.only_unmatched:
        taxonomy_subset = matching_service.get_unmatched_taxonomy(settings.similarity_threshold)
        target_ids = [tax.id for tax in taxonomy_subset]
        if not taxonomy_subset:
            click.echo("All taxonomy pages currently have matches above the semantic threshold.")
            return

    if taxonomy_subset is not None:
        click.echo(f"Targeting {len(taxonomy_subset)} taxonomy row(s) for this run")

    if options.force_semantic:
        deleted = db.clear_matching_results(target_ids, None)
        click.echo(f"Cleared {deleted} matching rows prior to semantic stage re-run")

    if options.force_llm:
        deleted_llm = db.clear_matching_results(
            target_ids,
            [
                MatchStage.LLM_CATEGORIZED,
                MatchStage.NEEDS_LLM_REVIEW,
                MatchStage.NEEDS_HUMAN_REVIEW,
            ],
        )
        click.echo(f"Cleared {deleted_llm} LLM/review rows prior to fallback stage")

    click.echo("Starting cascading matching workflow...")
    _describe_config(settings)

    stats = workflow_service.run_matching_workflow(
        taxonomy_pages=taxonomy_subset,
        content_items=content_subset,
        batch_mode=options.batch,
    )

    click.echo("\n=== Matching Results ===")
    click.echo(f"✓ Semantic matched: {stats['semantic_matched']}")
    click.echo(f"✓ LLM categorized: {stats['llm_categorized']}")
    click.echo(f"⚠ Needs review: {stats['needs_review']}")
    total = stats["semantic_matched"] + stats["llm_categorized"] + stats["needs_review"]
    click.echo(f"Total processed: {total}")
    if batch_ids := stats.get("llm_batch_ids"):
        click.echo(f"Batch job IDs: {', '.join(batch_ids)}")


def execute_full_run(options: FullRunOptions) -> None:
    settings = get_settings()
    db = SupabaseClient(settings)
    ingestion_service = IngestionService(settings, db)
    exporter = CSVExporter(db)

    csv_path = options.taxonomy_file or settings.taxonomy_file_path
    click.echo(f"[1/4] Loading taxonomy from {csv_path}...")
    loaded_count = ingestion_service.load_taxonomy_from_csv(csv_path)
    click.echo(f"    ✓ Loaded {loaded_count} taxonomy pages")

    default_sites = settings.get_wordpress_site_tokens()
    site_tokens = resolve_site_tokens_override(options.sites, default_sites)
    click.echo(f"[2/4] Ingesting content from {len(site_tokens)} site(s)...")
    if options.resume and options.since:
        click.echo(
            "    ⚠ Both --resume and --since provided; --since takes precedence for all sites."
        )
    ingested = ingestion_service.ingest_wordpress_sites(
        site_tokens,
        max_pages=options.max_pages,
        since=options.since,
        resume=options.resume,
    )
    click.echo(f"    ✓ Ingested {ingested} content items")

    overrides: dict[str, bool | float] = {}
    if options.skip_semantic:
        overrides["enable_semantic_matching"] = False
    if options.skip_llm:
        overrides["enable_llm_categorization"] = False
    if options.threshold is not None:
        overrides["similarity_threshold"] = options.threshold

    match_settings = get_settings(overrides=overrides or None)
    workflow_service = WorkflowService(match_settings, db)

    click.echo("[3/4] Running cascading matching workflow...")
    _describe_config(match_settings)
    stats = workflow_service.run_matching_workflow(batch_mode=options.batch)
    click.echo(
        "    ✓ Matching stats – "
        f"semantic: {stats['semantic_matched']}, LLM: {stats['llm_categorized']}, review: {stats['needs_review']}"
    )

    click.echo(f"[4/4] Exporting combined results to {options.output}...")
    row_count = exporter.export_to_csv(
        options.output,
        include_unmatched=True,
        min_similarity=options.min_similarity,
    )
    click.echo(f"    ✓ Exported {row_count} rows to {options.output}")


def execute_optimize_dataset(options: OptimizeDatasetOptions) -> None:
    if options.train_split is not None and not 0 < options.train_split < 1:
        raise click.BadParameter("Train split must be between 0 and 1", param_hint="--train-split")

    if options.optimizer.lower() == "gepa":
        budget_count = sum(
            bool(x) for x in [options.budget, options.max_full_evals, options.max_metric_calls]
        )
        if budget_count > 1:
            raise click.UsageError(
                "For GEPA optimizer, provide exactly one of: --budget, --max-full-evals, or --max-metric-calls"
            )

    overrides: dict[str, float | int] = {}
    if options.train_split is not None:
        overrides["dspy_train_split_ratio"] = options.train_split
    if options.seed is not None:
        overrides["dspy_optimization_seed"] = options.seed

    settings = get_settings(overrides=overrides or None)
    db = SupabaseClient(settings)
    optimizer_instance = DSPyOptimizer(settings, db)

    click.echo(f"Loading dataset from {options.dataset}...")
    try:
        training_data = optimizer_instance.load_training_dataset(options.dataset)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error loading dataset: {exc}", err=True)
        sys.exit(1)

    if len(training_data) < 10:
        click.echo(
            "Warning: Very few training examples. Results may not be optimal.",
            err=True,
        )
        click.echo("Recommend at least 50 examples, optimal for 300+ examples.")

    click.echo(
        f"Optimizing with {len(training_data)} examples using {options.optimizer} optimizer..."
    )
    click.echo(
        "⚠ This process is expensive by nature (multiple iterations, metric evaluations, LLM calls)."
    )

    optimizer_kwargs: dict[str, int | str | None] = {}
    if options.budget:
        optimizer_kwargs["budget"] = options.budget.lower()
    if options.max_full_evals is not None:
        optimizer_kwargs["max_full_evals"] = options.max_full_evals
    if options.max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = options.max_metric_calls
    if options.num_threads is not None:
        optimizer_kwargs["num_threads"] = options.num_threads
    if options.display_table is not None:
        optimizer_kwargs["display_table"] = options.display_table

    train_split_ratio = options.train_split or settings.dspy_train_split_ratio
    optimizer_config = {
        "optimizer": options.optimizer,
        "budget": options.budget,
        "max_full_evals": options.max_full_evals,
        "max_metric_calls": options.max_metric_calls,
        "train_split": train_split_ratio,
        "num_threads": options.num_threads or settings.dspy_num_threads,
        "seed": options.seed or settings.dspy_optimization_seed,
        "metric": settings.dspy_optimization_metric,
    }

    next_version = optimizer_instance._get_next_version("unused")
    output_path = options.output or (MODELS_DIR / f"matcher_v{next_version}.json")

    before_model = deepcopy(optimizer_instance.matcher)
    start_time = time.time()

    try:
        train_size = int(len(training_data) * train_split_ratio)
        val_size = len(training_data) - train_size

        optimized_model = optimizer_instance.optimize_with_dataset(
            training_data,
            optimizer_type=options.optimizer.lower(),
            **optimizer_kwargs,  # type: ignore[arg-type]
        )

        duration = time.time() - start_time

        validation_score = None
        try:
            from dspy.evaluate import Evaluate

            split_point = int(len(training_data) * train_split_ratio)
            val_set = training_data[split_point:]
            evaluator = Evaluate(
                devset=val_set,
                metric=optimizer_instance.metric_fn,
                num_threads=1,
                display_progress=False,
            )
            raw_score = evaluator(optimized_model)
            validation_score = _coerce_validation_score(raw_score)
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            click.echo(f"⚠ Warning: Could not compute validation score: {exc}", err=True)

        optimizer_instance.save_optimized_model(optimized_model, str(output_path))
        click.echo(f"✓ Optimization complete. Model saved to {output_path}")

        if not options.no_save_config:
            try:
                config_file = optimizer_instance.save_optimization_config(
                    before_model=before_model,
                    after_model=optimized_model,
                    optimizer_type=options.optimizer.lower(),
                    optimizer_config=optimizer_config,
                    training_size=train_size,
                    validation_size=val_size,
                    model_file_path=str(output_path),
                    validation_score=validation_score,
                    duration_seconds=duration,
                )
                click.echo(f"✓ Optimization config saved to {config_file.name}")
            except Exception as exc:  # pragma: no cover - logging only
                click.echo(f"⚠ Warning: Could not save optimization config: {exc}", err=True)

        if options.report:
            report_content = optimizer_instance.generate_optimization_report(
                before_model=before_model,
                after_model=optimized_model,
                optimizer_type=options.optimizer.lower(),
                optimizer_config=optimizer_config,
                training_size=train_size,
                validation_size=val_size,
                validation_score=validation_score,
                duration_seconds=duration,
            )
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(report_content, encoding="utf-8")
            click.echo(f"✓ Optimization report saved to {options.report}")

    except Exception as exc:  # pragma: no cover - forwarded to CLI for exit
        click.echo(f"Error during optimization: {exc}", err=True)
        raise


def _coerce_validation_score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    score_attr = getattr(value, "score", None)
    if isinstance(score_attr, (int, float)):
        return float(score_attr)
    summary_attr = getattr(value, "summary", None)
    if isinstance(summary_attr, dict):
        maybe_score = summary_attr.get("score")
        if isinstance(maybe_score, (int, float)):
            return float(maybe_score)
    return None
