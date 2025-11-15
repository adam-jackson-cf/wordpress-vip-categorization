"""Test script to verify setup and configuration."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.connectors.wordpress_vip import WordPressVIPConnector
from src.data.supabase_client import SupabaseClient
from src.services.matching import MatchingService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    try:
        settings = get_settings()
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  - Supabase URL: {settings.supabase_url}")
        logger.info(
            f"  - Semantic API: {settings.semantic_embedding_model} @ {settings.semantic_base_url}"
        )
        logger.info(f"  - LLM API: {settings.llm_model} @ {settings.llm_base_url}")
        return settings
    except Exception as e:
        logger.error(f"✗ Configuration failed: {e}")
        raise


def test_supabase_connection(settings):
    """Test Supabase connection."""
    logger.info("\nTesting Supabase connection...")
    try:
        db = SupabaseClient(settings)
        logger.info("✓ Supabase client initialized")

        # Try to query (will fail if tables don't exist, but connection is verified)
        try:
            content = db.get_all_content(limit=1)
            logger.info(f"✓ Supabase connection verified (found {len(content)} content items)")
        except Exception as e:
            logger.warning(
                "⚠ Tables may not exist yet. Please run src/data/schema.sql in Supabase SQL editor."
            )
            logger.warning(f"  Error: {e}")

        return db
    except Exception as e:
        logger.error(f"✗ Supabase connection failed: {e}")
        raise


def test_wordpress_connector(settings):
    """Test WordPress connector with WordPress.org news."""
    logger.info("\nTesting WordPress connector...")
    try:
        sites = settings.get_wordpress_sites()
        test_site = sites[0]
        logger.info(f"Testing with site: {test_site}")

        connector = WordPressVIPConnector(test_site)

        # Test connection
        if not connector.test_connection():
            raise Exception("Failed to connect to WordPress site")

        logger.info("✓ WordPress connection successful")

        # Fetch a few posts
        posts = connector.get_posts(page=1, per_page=3)
        logger.info(f"✓ Fetched {len(posts)} posts from {test_site}")

        if posts:
            logger.info(f"  Sample post: {posts[0].get('title', {}).get('rendered', 'N/A')}")

        return connector
    except Exception as e:
        logger.error(f"✗ WordPress connector failed: {e}")
        raise


def test_openrouter_embeddings(settings, db):
    """Test embeddings API for semantic matching."""
    logger.info("\nTesting semantic embeddings...")
    try:
        matching_service = MatchingService(settings, db)
        logger.info(
            f"✓ Matching service initialized with model: {matching_service.embedding_model}"
        )

        # Test embedding generation
        test_text = "WordPress is a free and open-source content management system."
        embedding = matching_service.get_embedding(test_text)

        logger.info(f"✓ Generated embedding with {len(embedding)} dimensions")
        logger.info(f"  Sample values: {embedding[:5]}")

        # Test similarity computation
        embedding2 = matching_service.get_embedding(
            "WordPress CMS is used for blogging and websites."
        )
        similarity = matching_service.compute_similarity(embedding, embedding2)
        logger.info(f"✓ Similarity score between related texts: {similarity:.4f}")

        return matching_service
    except Exception as e:
        logger.error(f"✗ OpenRouter embeddings failed: {e}")
        logger.error("  Note: Verify your semantic provider credentials and model access.")
        raise


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("WordPress VIP Categorization - Setup Test")
    logger.info("=" * 70)

    try:
        # Test 1: Configuration
        settings = test_configuration()

        # Test 2: Supabase
        db = test_supabase_connection(settings)

        # Test 3: WordPress Connector
        _connector = test_wordpress_connector(settings)

        # Test 4: OpenRouter Embeddings
        _matching_service = test_openrouter_embeddings(settings, db)

        logger.info("\n" + "=" * 70)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info(
            "1. Run the src/data/schema.sql file in your Supabase SQL Editor to create tables"
        )
        logger.info("2. Load taxonomy: python -m src.cli load-taxonomy")
        logger.info("3. Ingest content: python -m src.cli ingest")
        logger.info("4. Perform matching: python -m src.cli match")
        logger.info("5. Export results: python -m src.cli export --output results/results.csv")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("✗ SETUP TEST FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
