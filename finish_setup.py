"""Continue setup from where it left off - run matching and export."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.data.supabase_client import SupabaseClient
from src.services.matching import MatchingService
from src.exporters.csv_exporter import CSVExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run matching and export."""
    settings = get_settings()
    db = SupabaseClient(settings)

    logger.info("="*70)
    logger.info("WordPress VIP Categorization - Matching & Export")
    logger.info("="*70)

    # Check what we have
    content_count = len(db.get_all_content())
    taxonomy_count = len(db.get_all_taxonomy())

    logger.info(f"\n✓ Found {content_count} WordPress content items")
    logger.info(f"✓ Found {taxonomy_count} taxonomy pages")

    if content_count == 0:
        logger.error("No content found! Please run ingestion first.")
        return 1

    if taxonomy_count == 0:
        logger.error("No taxonomy found! Please load taxonomy first.")
        return 1

    # Run matching
    logger.info("\n" + "="*70)
    logger.info("Running Semantic Matching")
    logger.info("="*70)

    try:
        matching_service = MatchingService(settings, db)
        logger.info(f"Matching with threshold: {settings.similarity_threshold}")

        results = matching_service.match_all_taxonomy_batch(
            min_threshold=settings.similarity_threshold,
            store_results=True
        )

        matched_count = sum(1 for r in results.values() if r and r.content_id)
        total_count = len(results)

        logger.info(f"\n✓ Matched {matched_count}/{total_count} taxonomy pages")

        # Show sample matches
        if results:
            logger.info("\nTop matches:")
            sample_count = 0
            for taxonomy_id, result in results.items():
                if result and result.content_id and sample_count < 4:
                    taxonomy = db.get_taxonomy_by_id(taxonomy_id)
                    content = db.get_content_by_id(result.content_id)
                    if taxonomy and content:
                        logger.info(f"\n  {sample_count + 1}. {taxonomy.category}: {taxonomy.url}")
                        logger.info(f"     → {content.title}")
                        logger.info(f"     → {content.url}")
                        logger.info(f"     Score: {result.similarity_score:.4f}")
                        sample_count += 1

    except Exception as e:
        logger.error(f"Error during matching: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Export results
    logger.info("\n" + "="*70)
    logger.info("Exporting Results")
    logger.info("="*70)

    try:
        exporter = CSVExporter(db)

        # Export all results
        output_path = Path("results.csv")
        count = exporter.export_to_csv(output_path, include_unmatched=True)
        logger.info(f"✓ Exported {count} rows to {output_path}")

        # Export unmatched only
        unmatched_path = Path("unmatched.csv")
        unmatched_count = exporter.export_unmatched_only(unmatched_path)
        logger.info(f"✓ Exported {unmatched_count} unmatched items to {unmatched_path}")

    except Exception as e:
        logger.error(f"Error exporting: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Success!
    logger.info("\n" + "="*70)
    logger.info("✅ SETUP COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📊 Summary:")
    logger.info(f"  - WordPress content: {content_count} items")
    logger.info(f"  - Taxonomy pages: {taxonomy_count} items")
    logger.info(f"  - Matched: {matched_count}/{total_count}")
    logger.info(f"  - Unmatched: {unmatched_count}")

    logger.info("\n📁 Output files:")
    logger.info("  - results.csv (all matches)")
    logger.info("  - unmatched.csv (taxonomy pages without matches)")

    logger.info("\n💡 Next steps:")
    logger.info("  1. Open results.csv in your spreadsheet application")
    logger.info("  2. Filter by empty target_url to find unmatched items")
    logger.info("  3. Review similarity scores")
    logger.info("  4. Adjust keywords in taxonomy.csv if needed")
    logger.info("  5. Re-run matching with: python -m src.cli match")

    return 0


if __name__ == "__main__":
    sys.exit(main())
