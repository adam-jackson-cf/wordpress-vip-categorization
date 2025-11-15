"""Command-line interface for WordPress VIP categorization."""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

import click

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.exporters.csv_exporter import CSVExporter
from src.models import MatchStage
from src.optimization.dspy_optimizer import DSPyOptimizer
from src.optimization.evaluator import Evaluator
from src.services.categorization import CategorizationService
from src.services.ingestion import IngestionService
from src.services.matching import MatchingService
from src.services.workflow import WorkflowService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """WordPress VIP Content Categorization CLI."""
    pass


def _load_taxonomy_urls(csv_path: Path) -> list[str]:
    """Extract the `url` column from a taxonomy CSV file."""

    with open(csv_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "url" not in reader.fieldnames:
            raise click.ClickException("Expected 'url' column in taxonomy file")

        urls = [row["url"].strip() for row in reader if row.get("url")]

    return [url for url in urls if url]


@cli.command()
def init_db() -> None:
    """Initialize database schema."""
    click.echo("Database initialization should be done via Supabase dashboard.")
    click.echo("Please run the schema.sql file in your Supabase SQL editor.")
    click.echo("See: schema.sql")


@cli.command()
@click.option(
    "--sites",
    help="Comma-separated list of WordPress sites to ingest (overrides config)",
)
@click.option("--max-pages", type=int, help="Maximum pages to fetch per site")
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"]),
    help="Only ingest content published after this timestamp",
)
@click.option(
    "--resume/--no-resume", default=False, help="Resume per-site from last known published date"
)
def ingest(
    sites: str | None,
    max_pages: int | None,
    since: datetime | None,
    resume: bool,
) -> None:
    """Ingest content from WordPress sites."""
    settings = get_settings()
    db = SupabaseClient(settings)
    ingestion_service = IngestionService(settings, db)

    # Use provided sites or get from config
    site_urls = sites.split(",") if sites else settings.get_wordpress_sites()

    click.echo(f"Starting ingestion from {len(site_urls)} sites...")
    if resume and since:
        click.echo(
            "⚠ Both --resume and --since provided; --since will take precedence for all sites."
        )

    total = ingestion_service.ingest_wordpress_sites(
        site_urls,
        max_pages=max_pages,
        since=since,
        resume=resume,
    )
    click.echo(f"✓ Ingested {total} content items")


@cli.command()
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to taxonomy CSV file",
)
def load_taxonomy(taxonomy_file: Path | None) -> None:
    """Load taxonomy from CSV file."""
    settings = get_settings()
    db = SupabaseClient(settings)
    ingestion_service = IngestionService(settings, db)

    csv_path = taxonomy_file or settings.taxonomy_file_path

    click.echo(f"Loading taxonomy from {csv_path}...")
    count = ingestion_service.load_taxonomy_from_csv(csv_path)
    click.echo(f"✓ Loaded {count} taxonomy pages")


@cli.command()
@click.option("--batch/--no-batch", default=True, help="Use batch API (recommended)")
@click.option("--wait/--no-wait", default=True, help="Wait for batch completion")
def categorize(batch: bool, wait: bool) -> None:
    """Categorize content using OpenAI."""
    settings = get_settings()
    db = SupabaseClient(settings)
    categorization_service = CategorizationService(settings, db)

    if not batch:
        click.echo("Error: Non-batch categorization not implemented yet.")
        click.echo("Please use --batch flag.")
        sys.exit(1)

    # Get content and categories
    content_items = db.get_all_content()
    categories = categorization_service.get_categories_from_taxonomy()

    if not content_items:
        click.echo("Error: No content found. Run 'ingest' first.")
        sys.exit(1)

    if not categories:
        click.echo("Error: No categories found. Run 'load-taxonomy' first.")
        sys.exit(1)

    click.echo(f"Categorizing {len(content_items)} items into {len(categories)} categories...")

    batch_id = categorization_service.categorize_content_batch(content_items, categories, wait=wait)

    if wait:
        click.echo(f"✓ Categorization complete (batch: {batch_id})")
    else:
        click.echo(f"✓ Batch submitted: {batch_id}")
        click.echo(f"Check status with: python -m src.cli batch-status --id {batch_id}")


@cli.command()
@click.option("--id", "batch_id", required=True, help="Batch ID to check")
def batch_status(batch_id: str) -> None:
    """Check status of a batch job."""
    settings = get_settings()
    db = SupabaseClient(settings)
    categorization_service = CategorizationService(settings, db)

    status = categorization_service.get_batch_status(batch_id)

    click.echo(f"Batch ID: {status.batch_id}")
    click.echo(f"Status: {status.status}")
    click.echo(f"Created: {status.created_at}")
    click.echo(f"Completed: {status.completed_at or 'N/A'}")
    click.echo(
        f"Progress: {status.request_counts.get('completed', 0)}/"
        f"{status.request_counts.get('total', 0)}"
    )


@cli.command()
@click.option("--threshold", type=float, help="Minimum similarity threshold (default from config)")
@click.option("--batch/--no-batch", default=True, help="Use batch embedding mode")
@click.option("--skip-semantic", is_flag=True, help="Skip semantic matching stage")
@click.option("--skip-llm", is_flag=True, help="Skip LLM categorization stage")
@click.option(
    "--taxonomy-ids",
    help="Comma-separated list of taxonomy UUIDs to (re)process",
)
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True, path_type=Path),
    help="CSV containing taxonomy rows (uses 'url' column) to (re)process",
)
@click.option(
    "--only-unmatched",
    is_flag=True,
    help="Only reprocess taxonomy rows currently below the semantic threshold",
)
@click.option(
    "--force-semantic",
    is_flag=True,
    help="Clear previous matching rows (all stages) for targeted taxonomy before rerun",
)
@click.option(
    "--force-llm",
    is_flag=True,
    help="Clear previous LLM/human-review rows for targeted taxonomy before rerun",
)
def match(
    threshold: float | None,
    batch: bool,
    skip_semantic: bool,
    skip_llm: bool,
    taxonomy_ids: str | None,
    taxonomy_file: Path | None,
    only_unmatched: bool,
    force_semantic: bool,
    force_llm: bool,
) -> None:
    """Match taxonomy to content using cascading semantic + LLM workflow.

    Supports targeted reruns via --taxonomy-ids/--taxonomy-file or --only-unmatched.
    Use stage flags to disable/force semantic or LLM passes as needed.
    """
    filters_selected = sum(bool(flag) for flag in [taxonomy_ids, taxonomy_file, only_unmatched])
    if filters_selected > 1:
        raise click.UsageError("Use at most one of --taxonomy-ids/--taxonomy-file/--only-unmatched")

    settings = get_settings()
    db = SupabaseClient(settings)

    # Override config with CLI flags if provided
    if skip_semantic:
        settings.enable_semantic_matching = False
    if skip_llm:
        settings.enable_llm_categorization = False

    # Override semantic threshold if provided
    if threshold is not None:
        settings.similarity_threshold = threshold

    matching_service = MatchingService(settings, db)
    workflow_service = WorkflowService(settings, db, matching_service=matching_service)

    taxonomy_subset: list | None = None
    target_ids: list[UUID] | None = None

    if taxonomy_ids:
        try:
            parsed_ids = [UUID(value.strip()) for value in taxonomy_ids.split(",") if value.strip()]
        except ValueError as exc:
            raise click.BadParameter("Invalid UUID in --taxonomy-ids") from exc

        taxonomy_subset = db.get_taxonomy_by_ids(parsed_ids)
        target_ids = [tax.id for tax in taxonomy_subset]

        if not taxonomy_subset:
            click.echo("No taxonomy rows matched the provided IDs; nothing to do.")
            return

    elif taxonomy_file:
        urls = _load_taxonomy_urls(taxonomy_file)
        if not urls:
            click.echo("No URLs found in taxonomy file; nothing to do.")
            return

        taxonomy_subset = db.get_taxonomy_by_urls(urls)
        target_ids = [tax.id for tax in taxonomy_subset]

        if not taxonomy_subset:
            click.echo("None of the provided URLs exist in the database; aborting.")
            return

        missing_count = len(set(urls) - {str(tax.url) for tax in taxonomy_subset})
        if missing_count:
            click.echo(f"⚠ Skipping {missing_count} URL(s) not present in Supabase.")

    elif only_unmatched:
        taxonomy_subset = matching_service.get_unmatched_taxonomy(settings.similarity_threshold)
        target_ids = [tax.id for tax in taxonomy_subset]

        if not taxonomy_subset:
            click.echo("All taxonomy pages currently have matches above the semantic threshold.")
            return

    if taxonomy_subset is not None:
        click.echo(f"Targeting {len(taxonomy_subset)} taxonomy row(s) for this run")

    if force_semantic:
        deleted = db.clear_matching_results(target_ids, None)
        click.echo(f"Cleared {deleted} matching rows prior to semantic stage re-run")

    if force_llm:
        deleted_llm = db.clear_matching_results(
            target_ids,
            [MatchStage.LLM_CATEGORIZED, MatchStage.NEEDS_HUMAN_REVIEW],
        )
        click.echo(f"Cleared {deleted_llm} LLM/review rows prior to fallback stage")

    click.echo("Starting cascading matching workflow...")
    click.echo(
        f"- Semantic matching: {'enabled' if settings.enable_semantic_matching else 'disabled'} (threshold: {settings.similarity_threshold})"
    )
    click.echo(
        f"- LLM categorization: {'enabled' if settings.enable_llm_categorization else 'disabled'} (threshold: {settings.llm_confidence_threshold})"
    )

    stats = workflow_service.run_matching_workflow(
        taxonomy_pages=taxonomy_subset,
        batch_mode=batch,
    )

    click.echo("\n=== Matching Results ===")
    click.echo(f"✓ Semantic matched: {stats['semantic_matched']}")
    click.echo(f"✓ LLM categorized: {stats['llm_categorized']}")
    click.echo(f"⚠ Needs review: {stats['needs_review']}")
    click.echo(
        f"Total processed: {stats['semantic_matched'] + stats['llm_categorized'] + stats['needs_review']}"
    )


@cli.command(name="full-run")
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to taxonomy CSV file (defaults to config)",
)
@click.option(
    "--sites",
    help="Comma-separated list of WordPress sites to ingest (defaults to config)",
)
@click.option("--max-pages", type=int, help="Maximum pages to fetch per site during ingestion")
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"]),
    help="Only ingest content published after this timestamp",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume per-site ingestion from last known published date",
)
@click.option("--threshold", type=float, help="Minimum semantic similarity threshold")
@click.option(
    "--batch/--no-batch", default=True, help="Use batch embedding mode for semantic stage"
)
@click.option("--skip-semantic", is_flag=True, help="Skip semantic matching stage")
@click.option("--skip-llm", is_flag=True, help="Skip LLM categorization stage")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination CSV file for combined results",
)
@click.option(
    "--min-similarity",
    type=float,
    help="Optional post-filter on similarity when writing the CSV",
)
def full_run(
    taxonomy_file: Path | None,
    sites: str | None,
    max_pages: int | None,
    since: datetime | None,
    resume: bool,
    threshold: float | None,
    batch: bool,
    skip_semantic: bool,
    skip_llm: bool,
    output: Path,
    min_similarity: float | None,
) -> None:
    """Run taxonomy load → ingestion → matching → export with one command."""

    settings = get_settings()
    db = SupabaseClient(settings)

    ingestion_service = IngestionService(settings, db)
    exporter = CSVExporter(db)

    # Step 1: Load taxonomy
    csv_path = taxonomy_file or settings.taxonomy_file_path
    click.echo(f"[1/4] Loading taxonomy from {csv_path}...")
    loaded_count = ingestion_service.load_taxonomy_from_csv(csv_path)
    click.echo(f"    ✓ Loaded {loaded_count} taxonomy pages")

    # Step 2: Ingest content
    site_list = sites.split(",") if sites else settings.get_wordpress_sites()
    click.echo(f"[2/4] Ingesting content from {len(site_list)} site(s)...")
    if resume and since:
        click.echo(
            "    ⚠ Both --resume and --since provided; --since takes precedence for all sites."
        )
    ingested_count = ingestion_service.ingest_wordpress_sites(
        site_list,
        max_pages=max_pages,
        since=since,
        resume=resume,
    )
    click.echo(f"    ✓ Ingested {ingested_count} content items")

    # Step 3: Matching workflow with optional overrides
    if skip_semantic:
        settings.enable_semantic_matching = False
    if skip_llm:
        settings.enable_llm_categorization = False
    if threshold is not None:
        settings.similarity_threshold = threshold

    workflow_service = WorkflowService(settings, db)

    click.echo("[3/4] Running cascading matching workflow...")
    click.echo(
        f"    Semantic stage: {'enabled' if settings.enable_semantic_matching else 'disabled'} "
        f"(threshold={settings.similarity_threshold})"
    )
    click.echo(
        f"    LLM stage: {'enabled' if settings.enable_llm_categorization else 'disabled'} "
        f"(threshold={settings.llm_confidence_threshold})"
    )

    stats = workflow_service.run_matching_workflow(batch_mode=batch)
    click.echo(
        "    ✓ Matching stats – "
        f"semantic: {stats['semantic_matched']}, LLM: {stats['llm_categorized']}, review: {stats['needs_review']}"
    )

    # Step 4: Export combined CSV
    click.echo(f"[4/4] Exporting combined results to {output}...")
    row_count = exporter.export_to_csv(
        output,
        include_unmatched=True,
        min_similarity=min_similarity,
    )
    click.echo(f"    ✓ Exported {row_count} rows to {output}")


@cli.command()
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Output CSV path")
@click.option("--unmatched-only", is_flag=True, help="Export only unmatched taxonomy")
@click.option("--min-similarity", type=float, help="Minimum similarity threshold")
def export(output: Path, unmatched_only: bool, min_similarity: float | None) -> None:
    """Export matching results to CSV."""
    settings = get_settings()
    db = SupabaseClient(settings)
    exporter = CSVExporter(db)

    click.echo(f"Exporting results to {output}...")

    if unmatched_only:
        count = exporter.export_unmatched_only(output)
    else:
        count = exporter.export_to_csv(
            output, include_unmatched=True, min_similarity=min_similarity
        )

    click.echo(f"✓ Exported {count} rows to {output}")


@cli.command()
def stats() -> None:
    """Show statistics about ingested data."""
    settings = get_settings()
    db = SupabaseClient(settings)
    ingestion_service = IngestionService(settings, db)

    stats = ingestion_service.get_ingestion_stats()

    click.echo("=== Data Statistics ===")
    click.echo(f"WordPress Content: {stats['wordpress_content']}")
    click.echo(f"Taxonomy Pages: {stats['taxonomy_pages']}")
    click.echo(f"Matching Results: {stats['matchings']}")


@cli.command()
def evaluate() -> None:
    """Evaluate categorization and matching quality."""
    settings = get_settings()
    db = SupabaseClient(settings)
    evaluator = Evaluator(db)

    results = evaluator.evaluate_all()

    click.echo("=== Evaluation Results ===")
    click.echo("\nCategorization:")
    for key, value in results["categorization"].items():
        click.echo(f"  {key}: {value}")

    click.echo("\nMatching:")
    for key, value in results["matching"].items():
        click.echo(f"  {key}: {value}")


@cli.command()
@click.option("--num-trials", type=int, help="Number of optimization trials")
def optimize_prompts(num_trials: int | None) -> None:
    """Optimize categorization prompts using DSPy."""
    settings = get_settings()
    db = SupabaseClient(settings)
    optimizer = DSPyOptimizer(settings, db)

    # Get content with existing categorizations for training
    content_items = db.get_all_content()
    categories = db.get_all_taxonomy()

    if not content_items or not categories:
        click.echo(
            "Error: Need both content and taxonomy data. Run ingest and load-taxonomy first."
        )
        sys.exit(1)

    click.echo("Preparing training data...")
    training_data = optimizer.prepare_training_data(content_items, [c.category for c in categories])

    if len(training_data) < 10:
        click.echo("Warning: Very few training examples. Results may not be optimal.")

    click.echo(f"Optimizing with {len(training_data)} examples...")
    optimized_model = optimizer.optimize(training_data, max_labeled_demos=num_trials or 8)

    # Save optimized model
    model_path = "optimized_categorizer.json"
    optimizer.save_optimized_model(optimized_model, model_path)

    click.echo(f"✓ Optimization complete. Model saved to {model_path}")


if __name__ == "__main__":
    cli()
