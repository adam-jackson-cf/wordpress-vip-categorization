"""Complete setup script - initializes database and runs workflow."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.exporters.csv_exporter import CSVExporter
from src.services.ingestion import IngestionService
from src.services.matching import MatchingService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_database_tables(db):
    """Check if database tables exist."""
    logger.info("\n" + "=" * 70)
    logger.info("Checking database tables...")
    logger.info("=" * 70)

    try:
        # Try to query each table
        tables_ok = True

        try:
            db.get_all_content(limit=1)
            logger.info("✓ Table 'wordpress_content' exists")
        except Exception as e:
            logger.error(f"✗ Table 'wordpress_content' missing or inaccessible: {e}")
            tables_ok = False

        try:
            db.get_all_taxonomy()
            logger.info("✓ Table 'taxonomy_pages' exists")
        except Exception as e:
            logger.error(f"✗ Table 'taxonomy_pages' missing or inaccessible: {e}")
            tables_ok = False

        try:
            db.get_all_matchings()
            logger.info("✓ Table 'matching_results' exists")
        except Exception as e:
            logger.error(f"✗ Table 'matching_results' missing or inaccessible: {e}")
            tables_ok = False

        if not tables_ok:
            logger.error("\n" + "=" * 70)
            logger.error("❌ DATABASE TABLES MISSING")
            logger.error("=" * 70)
            logger.error("\nPlease initialize the database:")
            logger.error("1. Go to: https://supabase.com/dashboard/project/resciqkhyvnaxqpzabtb")
            logger.error("2. Click 'SQL Editor' in the left sidebar")
            logger.error("3. Click 'New Query'")
            logger.error("4. Copy the contents of 'schema.sql'")
            logger.error("5. Paste and click 'RUN'")
            logger.error("\nThen run this script again.")
            return False

        logger.info("\n✓ All required database tables exist!")
        return True

    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False


def load_taxonomy(settings, db):
    """Load taxonomy from CSV."""
    logger.info("\n" + "=" * 70)
    logger.info("Loading Taxonomy")
    logger.info("=" * 70)

    try:
        ingestion_service = IngestionService(settings, db)
        taxonomy_path = Path("data/taxonomy.csv")

        if not taxonomy_path.exists():
            logger.error(f"Taxonomy file not found: {taxonomy_path}")
            return False

        count = ingestion_service.load_taxonomy_from_csv(taxonomy_path)
        logger.info(f"✓ Loaded {count} taxonomy pages")
        return True

    except Exception as e:
        logger.error(f"Error loading taxonomy: {e}")
        return False


def ingest_content(settings, db):
    """Ingest WordPress content."""
    logger.info("\n" + "=" * 70)
    logger.info("Ingesting WordPress Content")
    logger.info("=" * 70)

    try:
        ingestion_service = IngestionService(settings, db)
        sites = settings.get_wordpress_sites()

        logger.info(f"Ingesting from: {sites}")
        logger.info("Limiting to 2 pages (first 10-20 posts) for testing...")

        count = ingestion_service.ingest_wordpress_sites(sites, max_pages=2)
        logger.info(f"✓ Ingested {count} content items")
        return True

    except Exception as e:
        logger.error(f"Error ingesting content: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_matching(settings, db):
    """Run semantic matching."""
    logger.info("\n" + "=" * 70)
    logger.info("Running Semantic Matching")
    logger.info("=" * 70)

    try:
        matching_service = MatchingService(settings, db)

        logger.info(f"Matching with threshold: {settings.similarity_threshold}")
        results = matching_service.match_all_taxonomy_batch(
            min_threshold=settings.similarity_threshold, store_results=True
        )

        matched_count = sum(1 for r in results.values() if r and r.content_id)
        total_count = len(results)

        logger.info(f"✓ Matched {matched_count}/{total_count} taxonomy pages")

        # Show some results
        if results:
            logger.info("\nSample matches:")
            for i, (taxonomy_id, result) in enumerate(list(results.items())[:3]):
                if result and result.content_id:
                    taxonomy = db.get_taxonomy_by_id(taxonomy_id)
                    content = db.get_content_by_id(result.content_id)
                    if taxonomy and content:
                        logger.info(f"  {i+1}. {taxonomy.url}")
                        logger.info(f"     → {content.url}")
                        logger.info(f"     Score: {result.similarity_score:.4f}")

        return True

    except Exception as e:
        logger.error(f"Error running matching: {e}")
        import traceback

        traceback.print_exc()
        return False


def export_results(db):
    """Export results to CSV."""
    logger.info("\n" + "=" * 70)
    logger.info("Exporting Results")
    logger.info("=" * 70)

    try:
        exporter = CSVExporter(db)
        output_path = Path("results.csv")

        count = exporter.export_to_csv(output_path, include_unmatched=True)
        logger.info(
            "✓ Exported %s rows to %s (filter blank target_url for items needing review)",
            count,
            output_path,
        )

        return True

    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return False


def main():
    """Run complete setup workflow."""
    logger.info("=" * 70)
    logger.info("WordPress VIP Categorization - Complete Setup")
    logger.info("=" * 70)

    try:
        # Load configuration
        settings = get_settings()
        logger.info("✓ Configuration loaded")
        logger.info("  - Supabase: %s", settings.supabase_url)
        logger.info(
            "  - Semantic embeddings: %s @ %s",
            settings.semantic_embedding_model,
            settings.semantic_base_url,
        )
        logger.info(
            "  - LLM model: %s @ %s",
            settings.llm_model,
            settings.llm_base_url,
        )

        # Initialize database client
        db = SupabaseClient(settings)
        logger.info("✓ Database client initialized")

        # Step 1: Check if tables exist
        if not check_database_tables(db):
            return 1

        # Step 2: Load taxonomy
        if not load_taxonomy(settings, db):
            return 1

        # Step 3: Ingest content
        if not ingest_content(settings, db):
            return 1

        # Step 4: Run matching
        if not run_matching(settings, db):
            return 1

        # Step 5: Export results
        if not export_results(db):
            return 1

        # Success!
        logger.info("\n" + "=" * 70)
        logger.info("✅ SETUP COMPLETE!")
        logger.info("=" * 70)
        logger.info("\nResults exported to:")
        logger.info("  - results.csv (all taxonomy rows; blank target_url => needs review)")
        logger.info("\nYou can now:")
        logger.info("  1. Open results.csv in a spreadsheet")
        logger.info("  2. Filter by empty target_url or match_stage == needs_human_review")
        logger.info("  3. Review similarity scores and adjust threshold if needed")
        logger.info("  4. Run more ingestion with: python -m src.cli ingest --sites <url>")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
