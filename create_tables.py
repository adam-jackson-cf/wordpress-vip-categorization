"""Create database tables using Supabase REST API."""

import logging
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Create database tables."""
    settings = get_settings()

    # Read schema file
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    logger.info("Creating database tables via Supabase SQL API...")

    # Use Supabase's SQL endpoint (requires service_role key)
    # Format: https://<project-ref>.supabase.co/rest/v1/rpc/<function-name>

    # Extract project ref from URL
    project_ref = settings.supabase_url.split('//')[1].split('.')[0]

    # SQL endpoint
    sql_url = f"https://{project_ref}.supabase.co/rest/v1/rpc/exec_sql"

    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Content-Type": "application/json"
    }

    try:
        # Try using direct SQL execution (may not be available)
        with httpx.Client() as client:
            response = client.post(
                sql_url,
                headers=headers,
                json={"query": schema_sql},
                timeout=30.0
            )

            if response.status_code == 200:
                logger.info("✓ Tables created successfully!")
                return 0
            else:
                logger.warning(f"Direct SQL execution not available: {response.status_code}")

    except Exception as e:
        logger.debug(f"API approach failed: {e}")

    # Alternative: Use curl command
    logger.info("\nAttempting via curl...")

    import subprocess

    # Write SQL to temp file
    temp_sql = Path("/tmp/schema.sql")
    temp_sql.write_text(schema_sql)

    # Try using psql if available (won't work in this environment)
    # Instead, provide clear instructions

    logger.info("\n" + "="*70)
    logger.info("DATABASE INITIALIZATION REQUIRED")
    logger.info("="*70)
    logger.info("\nPlease follow these steps:")
    logger.info("\n1. Go to your Supabase project:")
    logger.info("   https://supabase.com/dashboard/project/resciqkhyvnaxqpzabtb")
    logger.info("\n2. Click 'SQL Editor' in the left sidebar")
    logger.info("\n3. Click 'New Query'")
    logger.info("\n4. Copy and paste this SQL:\n")

    # Print the schema for easy copy-paste
    print("-" * 70)
    print(schema_sql)
    print("-" * 70)

    logger.info("\n5. Click 'RUN' (or press Ctrl+Enter)")
    logger.info("\n6. Wait for success message")
    logger.info("\n7. Run: python run_setup.py")

    return 1


if __name__ == "__main__":
    sys.exit(main())
