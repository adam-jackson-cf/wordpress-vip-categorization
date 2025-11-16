"""Command-line interface for WordPress VIP categorization."""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import click
import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.exporters.csv_exporter import CSVExporter
from src.models import MatchStage
from src.optimization.dspy_optimizer import MODELS_DIR, DSPyOptimizer
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

_sql_retryer = Retrying(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


def _post_schema_sql_with_retry(url: str, headers: dict[str, str], query: str) -> httpx.Response:
    """Execute Supabase SQL RPC with retry/backoff."""

    return _sql_retryer(
        lambda: httpx.post(
            url,
            headers=headers,
            json={"query": query},
            timeout=30.0,
        )
    )


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


@cli.command(name="init-db")
def init_db() -> None:
    """Initialize database schema from schema.sql.

    Attempts to execute the schema via Supabase's SQL RPC endpoint using your
    configured `SUPABASE_URL` and `SUPABASE_KEY`. If direct execution is not
    available, the SQL is printed with clear instructions for manual execution
    in the Supabase SQL editor.
    """
    settings = get_settings()
    schema_path = Path(__file__).resolve().parent / "data" / "schema.sql"

    if not schema_path.exists():
        raise click.ClickException(f"schema.sql not found at {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    click.echo("Initializing database schema from schema.sql...")
    click.echo("Attempting to execute via Supabase SQL RPC endpoint...")

    # Derive project ref from Supabase URL: https://<project-ref>.supabase.co/...
    try:
        project_ref = settings.supabase_url.split("//", maxsplit=1)[1].split(".", maxsplit=1)[0]
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Unable to derive project ref from SUPABASE_URL: {exc}"
        ) from exc

    sql_url = f"https://{project_ref}.supabase.co/rest/v1/rpc/exec_sql"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json",
    }

    try:
        response = _post_schema_sql_with_retry(sql_url, headers, schema_sql)

        if response.status_code == 200:
            click.echo("✓ Tables created successfully via Supabase SQL RPC.")
            return

        click.echo(
            f"⚠ Direct SQL execution via RPC returned status {response.status_code}; "
            "falling back to manual instructions."
        )
    except Exception as exc:  # pragma: no cover - network/remote failure
        click.echo(f"⚠ RPC-based SQL execution failed: {exc}")
        click.echo("Falling back to manual instructions.")

    # Manual fallback: print clear instructions and schema for copy-paste
    click.echo("\n" + "=" * 70)
    click.echo("DATABASE INITIALIZATION REQUIRED")
    click.echo("=" * 70)
    click.echo("\n1. Open your Supabase project dashboard.")
    click.echo("2. Click 'SQL Editor' in the left sidebar.")
    click.echo("3. Click 'New Query'.")
    click.echo("4. Copy and paste the following SQL, then click 'RUN':\n")
    click.echo("-" * 70)
    click.echo(schema_sql)
    click.echo("-" * 70)
    click.echo("\nOnce the schema has been applied, you can run:")
    click.echo("  python -m src.cli full-run --output results/results.csv")


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

    overrides: dict[str, bool | float] = {}
    if skip_semantic:
        overrides["enable_semantic_matching"] = False
    if skip_llm:
        overrides["enable_llm_categorization"] = False

    # Override semantic threshold if provided
    if threshold is not None:
        overrides["similarity_threshold"] = threshold

    settings = get_settings(overrides=overrides or None)
    db = SupabaseClient(settings)

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
        f"- LLM categorization: {'enabled' if settings.enable_llm_categorization else 'disabled'} (rubric-gated)"
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

    base_settings = get_settings()
    db = SupabaseClient(base_settings)

    ingestion_service = IngestionService(base_settings, db)
    exporter = CSVExporter(db)

    # Step 1: Load taxonomy
    csv_path = taxonomy_file or base_settings.taxonomy_file_path
    click.echo(f"[1/4] Loading taxonomy from {csv_path}...")
    loaded_count = ingestion_service.load_taxonomy_from_csv(csv_path)
    click.echo(f"    ✓ Loaded {loaded_count} taxonomy pages")

    # Step 2: Ingest content
    site_list = sites.split(",") if sites else base_settings.get_wordpress_sites()
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
    overrides: dict[str, bool | float] = {}
    if skip_semantic:
        overrides["enable_semantic_matching"] = False
    if skip_llm:
        overrides["enable_llm_categorization"] = False
    if threshold is not None:
        overrides["similarity_threshold"] = threshold

    match_settings = get_settings(overrides=overrides or None)

    workflow_service = WorkflowService(match_settings, db)

    click.echo("[3/4] Running cascading matching workflow...")
    click.echo(
        f"    Semantic stage: {'enabled' if match_settings.enable_semantic_matching else 'disabled'} "
        f"(threshold={match_settings.similarity_threshold})"
    )
    click.echo(
        f"    LLM stage: {'enabled' if match_settings.enable_llm_categorization else 'disabled'} (rubric-gated)"
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


@cli.command(name="evaluate-matcher")
@click.option(
    "--matcher-path",
    type=click.Path(path_type=Path),
    help="Optional path to a matcher JSON to load before evaluation (defaults to latest if present)",
)
def evaluate_matcher(matcher_path: Path | None) -> None:
    """Evaluate current database results and (optionally) load a matcher for downstream runs."""
    settings = get_settings()
    db = SupabaseClient(settings)

    if matcher_path:
        try:
            optimizer = DSPyOptimizer(settings, db)
            optimizer.load_optimized_model(str(matcher_path))
            click.echo(f"Loaded matcher from {matcher_path} for subsequent runs")
        except Exception as exc:
            click.echo(f"⚠ Warning: Could not load matcher {matcher_path}: {exc}", err=True)

    evaluator = Evaluator(db)
    results = evaluator.evaluate_all()

    click.echo("=== Matcher Evaluation (DB snapshot) ===")
    click.echo(f"- Categorization: {results['categorization']}")
    click.echo(f"- Matching: {results['matching']}")
    click.echo(f"- Rubric: {results['rubric']}")


@cli.command(name="compare-matchers")
@click.option(
    "--baseline",
    type=click.Path(path_type=Path),
    required=False,
    help="Baseline matcher path (omit to use current latest)",
)
@click.option(
    "--candidate",
    type=click.Path(path_type=Path),
    required=True,
    help="Candidate matcher path to compare against baseline",
)
def compare_matchers(baseline: Path | None, candidate: Path) -> None:
    """Compare baseline vs candidate by reporting current DB evaluation before and after load."""
    settings = get_settings()
    db = SupabaseClient(settings)
    evaluator = Evaluator(db)

    # Baseline snapshot
    base_metrics = evaluator.evaluate_all()
    click.echo("=== Baseline (before) ===")
    click.echo(f"- Categorization: {base_metrics['categorization']}")
    click.echo(f"- Matching: {base_metrics['matching']}")
    click.echo(f"- Rubric: {base_metrics['rubric']}")

    # Load candidate
    try:
        optimizer = DSPyOptimizer(settings, db)
        optimizer.load_optimized_model(str(candidate))
        click.echo(f"\nLoaded candidate matcher from {candidate}")
    except Exception as exc:
        click.echo(f"Error: Could not load candidate matcher {candidate}: {exc}", err=True)
        sys.exit(1)

    # Candidate snapshot (note: metrics reflect DB state; use workflow runs to generate fresh data)
    cand_metrics = evaluator.evaluate_all()
    click.echo("\n=== Candidate (after load) ===")
    click.echo(f"- Categorization: {cand_metrics['categorization']}")
    click.echo(f"- Matching: {cand_metrics['matching']}")
    click.echo(f"- Rubric: {cand_metrics['rubric']}")

    # Simple deltas for quick read
    def _delta(a: float, b: float) -> float:
        return round(b - a, 4)

    click.echo("\n=== Delta (candidate - baseline) ===")
    click.echo(
        f"- Match rate Δ: {_delta(base_metrics['matching'].get('match_rate', 0.0), cand_metrics['matching'].get('match_rate', 0.0))}"
    )
    click.echo(
        f"- Avg similarity Δ: {_delta(base_metrics['matching'].get('avg_similarity', 0.0), cand_metrics['matching'].get('avg_similarity', 0.0))}"
    )


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
    """Quick optimization using BootstrapFewShot (uses existing matching results from database).

    For thorough dataset-based optimization, use 'optimize-dataset' command instead.
    """
    settings = get_settings()
    db = SupabaseClient(settings)
    optimizer = DSPyOptimizer(settings, db)

    # Get content items for training data preparation
    content_items = db.get_all_content()

    if not content_items:
        click.echo("Error: No content found. Run 'ingest' first.")
        sys.exit(1)

    click.echo("Preparing training data from matching results...")
    training_data = optimizer.prepare_training_data(content_items)

    if len(training_data) < 10:
        click.echo("Warning: Very few training examples. Results may not be optimal.")
        click.echo("Run 'match' command first to generate matching results for training.")

    if not training_data:
        click.echo(
            "Error: No training data available. "
            "Run 'match' command first to generate matching results."
        )
        sys.exit(1)

    click.echo(f"Optimizing with {len(training_data)} examples...")
    optimized_model = optimizer.optimize(training_data, max_labeled_demos=num_trials or 8)

    # Determine next version and save optimized model
    next_version = optimizer._get_next_version("unused")
    model_path = MODELS_DIR / f"matcher_v{next_version}.json"
    optimizer.save_optimized_model(optimized_model, str(model_path))

    click.echo(f"✓ Optimization complete. Model saved to {model_path}")


@cli.command(name="optimize-dataset")
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to CSV or JSON training dataset file",
)
@click.option(
    "--optimizer",
    type=click.Choice(["gepa", "bootstrap-random-search", "bootstrap"], case_sensitive=False),
    default="gepa",
    help="Optimizer to use (default: gepa)",
)
@click.option(
    "--budget",
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    help="GEPA budget level (ignored if --max-full-evals or --max-metric-calls provided)",
)
@click.option(
    "--max-full-evals",
    type=int,
    help="Explicit GEPA budget: maximum full evaluations",
)
@click.option(
    "--max-metric-calls",
    type=int,
    help="Explicit GEPA budget: maximum metric calls",
)
@click.option(
    "--train-split",
    type=float,
    help="Training set split ratio (default: 0.2 for prompt optimizers)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Optional path to save optimized model (default: prompt-optimiser/models/matcher_vN.json)",
)
@click.option(
    "--save-program",
    is_flag=True,
    help="Save entire program structure (for complex optimizations)",
)
@click.option(
    "--num-threads",
    type=int,
    help="Parallel evaluation threads (default: 1, recommend 4-24 for production)",
)
@click.option(
    "--display-table",
    type=int,
    help="Number of example results to display (default: 5, 0 disables)",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    help="Path to save optimization report markdown file (optional)",
)
@click.option(
    "--no-save-config",
    is_flag=True,
    default=False,
    help="Disable automatic saving of optimization config JSON file",
)
def optimize_dataset(
    dataset: Path,
    optimizer: str,
    budget: str | None,
    max_full_evals: int | None,
    max_metric_calls: int | None,
    train_split: float | None,
    output: Path,
    save_program: bool,
    num_threads: int | None,
    display_table: int | None,
    seed: int | None,
    report: Path | None,
    no_save_config: bool,
) -> None:
    """Optimize prompts using dataset-based optimization (expensive by nature).

    This command performs thorough optimization using a provided dataset file.
    The process is expensive due to multiple iterations, metric evaluations, and LLM calls.

    Dataset format:
    - CSV: Must contain columns: taxonomy_category, taxonomy_description, content_summaries, best_match_index
    - JSON: Array of objects with same fields as CSV columns

    Example budget costs (GEPA):
    - Light: ~500-1600 metric calls, 6-12 full evaluations
    - Medium: Moderate metric calls and generations
    - Heavy: Comprehensive search with higher cost
    """
    # Validate train split early so argument errors trigger before loading settings/env vars
    if train_split is not None and not 0 < train_split < 1:
        raise click.BadParameter("Train split must be between 0 and 1", param_hint="--train-split")

    # Validate budget options (exactly one must be provided for GEPA)
    if optimizer.lower() == "gepa":
        budget_count = sum(bool(x) for x in [budget, max_full_evals, max_metric_calls])
        if budget_count == 0:
            # Use default from settings
            pass
        elif budget_count > 1:
            raise click.UsageError(
                "For GEPA optimizer, provide exactly one of: --budget, --max-full-evals, or --max-metric-calls"
            )

    overrides: dict[str, float | int] = {}
    if train_split is not None:
        overrides["dspy_train_split_ratio"] = train_split
    if seed is not None:
        overrides["dspy_optimization_seed"] = seed

    settings = get_settings(overrides=overrides or None)
    db = SupabaseClient(settings)
    optimizer_instance = DSPyOptimizer(settings, db)

    click.echo(f"Loading dataset from {dataset}...")
    try:
        training_data = optimizer_instance.load_training_dataset(dataset)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        sys.exit(1)

    if len(training_data) < 10:
        click.echo(
            "Warning: Very few training examples. Results may not be optimal.",
            err=True,
        )
        click.echo("Recommend at least 50 examples, optimal for 300+ examples.")

    click.echo(f"Optimizing with {len(training_data)} examples using {optimizer} optimizer...")
    click.echo(
        "⚠ This process is expensive by nature (multiple iterations, metric evaluations, LLM calls)."
    )

    # Prepare optimizer kwargs (typed for mypy)
    optimizer_kwargs: dict[str, int | str | None] = {}
    if budget:
        optimizer_kwargs["budget"] = budget.lower()
    if max_full_evals is not None:
        optimizer_kwargs["max_full_evals"] = max_full_evals
    if max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = max_metric_calls
    if num_threads is not None:
        optimizer_kwargs["num_threads"] = num_threads
    if display_table is not None:
        optimizer_kwargs["display_table"] = display_table

    # Build optimizer config dict for reporting/config saving
    train_split_ratio = train_split or settings.dspy_train_split_ratio
    optimizer_config = {
        "optimizer": optimizer,
        "budget": budget,
        "max_full_evals": max_full_evals,
        "max_metric_calls": max_metric_calls,
        "train_split": train_split_ratio,
        "num_threads": num_threads or settings.dspy_num_threads,
        "seed": seed or settings.dspy_optimization_seed,
    }

    # Determine version for this run and capture before state
    import time

    # Create a copy of the initial model for comparison
    from copy import deepcopy

    next_version = optimizer_instance._get_next_version("unused")
    if output is None:
        output = MODELS_DIR / f"matcher_v{next_version}.json"

    before_model = deepcopy(optimizer_instance.matcher)
    start_time = time.time()

    try:
        # Calculate train/val split for report
        train_size = int(len(training_data) * train_split_ratio)
        val_size = len(training_data) - train_size

        optimized_model = optimizer_instance.optimize_with_dataset(
            training_data,
            optimizer_type=optimizer.lower(),
            **optimizer_kwargs,  # type: ignore[arg-type]
        )

        duration = time.time() - start_time

        # Evaluate to get validation score
        validation_score_raw: float | None = None
        try:
            from dspy.evaluate import Evaluate

            split_point = int(len(training_data) * train_split_ratio)
            val_set = training_data[split_point:]
            evaluator = Evaluate(
                devset=val_set,
                metric=optimizer_instance.accuracy_metric,
                num_threads=1,
                display_progress=False,
            )
            validation_score_raw = evaluator(optimized_model)
        except Exception as e:
            logger.warning(f"Could not compute validation score: {e}")

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

        validation_score = _coerce_validation_score(validation_score_raw)

        # Save optimized model
        optimizer_instance.save_optimized_model(optimized_model, str(output))
        click.echo(f"✓ Optimization complete. Model saved to {output}")

        # Save optimization config (unless disabled)
        if not no_save_config:
            try:
                config_file = optimizer_instance.save_optimization_config(
                    before_model=before_model,
                    after_model=optimized_model,
                    optimizer_type=optimizer.lower(),
                    optimizer_config=optimizer_config,
                    training_size=train_size,
                    validation_size=val_size,
                    model_file_path=str(output),
                    validation_score=validation_score,
                    duration_seconds=duration,
                )
                click.echo(f"✓ Optimization config saved to {config_file.name}")
            except Exception as e:
                logger.warning(f"Could not save optimization config: {e}")
                click.echo(f"⚠ Warning: Could not save optimization config: {e}", err=True)

        # Generate report if requested
        if report:
            report_content = optimizer_instance.generate_optimization_report(
                before_model=before_model,
                after_model=optimized_model,
                optimizer_type=optimizer.lower(),
                optimizer_config=optimizer_config,
                training_size=train_size,
                validation_size=val_size,
                validation_score=validation_score,
                duration_seconds=duration,
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(report_content, encoding="utf-8")
            click.echo(f"✓ Optimization report saved to {report}")

    except Exception as e:
        click.echo(f"Error during optimization: {e}", err=True)
        logger.exception("Optimization failed")
        sys.exit(1)


@cli.group()
def workflow() -> None:
    """Workflow run management commands."""
    pass


@workflow.command(name="start")
@click.option("--run-key", help="Unique identifier for the workflow run")
@click.option("--batch/--no-batch", default=True, help="Toggle batch mode")
def workflow_start(run_key: str | None, batch: bool) -> None:
    """Start a managed workflow run with persisted checkpoints."""

    settings = get_settings()
    db = SupabaseClient(settings)
    workflow_service = WorkflowService(settings, db)

    key = run_key or f"run-{uuid4().hex}"
    stats = workflow_service.run_managed_workflow(run_key=key, batch_mode=batch, resume=False)
    click.echo(f"Workflow '{key}' completed: {stats}")


@workflow.command(name="resume")
@click.argument("run_key")
@click.option("--batch/--no-batch", default=True, help="Toggle batch mode for remaining stages")
def workflow_resume(run_key: str, batch: bool) -> None:
    """Resume an existing workflow run by run key."""

    settings = get_settings()
    db = SupabaseClient(settings)
    workflow_service = WorkflowService(settings, db)

    stats = workflow_service.run_managed_workflow(run_key=run_key, batch_mode=batch, resume=True)
    click.echo(f"Workflow '{run_key}' resumed with stats: {stats}")


@workflow.command(name="status")
@click.option("--run-key", help="Specific run key to inspect")
@click.option("--limit", default=10, help="Number of recent runs to list")
def workflow_status(run_key: str | None, limit: int) -> None:
    """Show workflow run metadata."""

    settings = get_settings()
    db = SupabaseClient(settings)

    if run_key:
        run = db.get_workflow_run_by_key(run_key)
        if not run:
            click.echo(f"No workflow run found with key '{run_key}'")
            return
        click.echo(
            f"Run {run.run_key}: status={run.status}, stage={run.current_stage}, stats={run.stats}"
        )
        return

    runs = db.list_workflow_runs(limit=limit)
    if not runs:
        click.echo("No workflow runs found")
        return

    for run in runs:
        click.echo(
            f"{run.run_key}: status={run.status}, stage={run.current_stage}, "
            f"started={run.started_at}, completed={run.completed_at}"
        )


if __name__ == "__main__":
    cli()
