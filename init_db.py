"""Initialize Supabase database schema."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database schema."""
    settings = get_settings()

    # Read schema file
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    # Connect to Supabase
    client = create_client(settings.supabase_url, settings.supabase_key)

    # Split into individual statements and execute
    # Note: Supabase REST API doesn't execute raw SQL
    # We'll use the PostgreSQL connection instead
    logger.info("Executing database schema...")

    try:
        # Execute the schema using Supabase's RPC
        # Since we can't execute DDL via REST API, we'll create a simpler approach
        # Let's use the SQL commands one by one

        statements = [s.strip() for s in schema_sql.split(';') if s.strip() and not s.strip().startswith('--')]

        for i, statement in enumerate(statements):
            if not statement:
                continue
            try:
                # Use rpc to execute SQL (this requires a stored procedure in Supabase)
                # Since that's not available, we'll inform the user
                logger.info(f"Statement {i+1}/{len(statements)}")
            except Exception as e:
                logger.debug(f"Statement execution note: {e}")

        logger.info("\n" + "="*70)
        logger.warning("⚠ Direct SQL execution via Python is not supported by Supabase REST API")
        logger.info("="*70)
        logger.info("\nPlease run the schema manually:")
        logger.info("1. Go to https://supabase.com/dashboard/project/resciqkhyvnaxqpzabtb")
        logger.info("2. Click 'SQL Editor' in the left sidebar")
        logger.info("3. Click 'New Query'")
        logger.info("4. Copy the contents of schema.sql")
        logger.info("5. Paste and click 'RUN'")
        logger.info("\nAfter that, run: python test_setup.py")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
