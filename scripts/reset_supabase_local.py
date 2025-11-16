"""Local helper to reset Supabase schema and data for development.

This script is intended for local environments. It prints clear instructions
and can optionally invoke the CLI init-db command to reapply schema.sql.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).parent.parent
    schema_path = project_root / "src" / "data" / "schema.sql"

    print("=== Supabase Local Reset Helper ===")
    print("\nThis helper will:")
    print("1) Reapply src/data/schema.sql using the CLI (service role required)")
    print("2) Optionally guide manual reset if RPC is unavailable")

    if not schema_path.exists():
        print(f"\nError: schema.sql not found at {schema_path}")
        return

    print("\nAttempting to run: python -m src.cli init-db")
    try:
        subprocess.run(
            ["python", "-m", "src.cli", "init-db"],
            cwd=str(project_root),
            check=True,
        )
        print("\n✓ Schema re-applied successfully via CLI.")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"\n⚠ CLI init-db failed: {exc}")
        print("Falling back to manual instructions:")
        print("\n1. Open Supabase SQL Editor")
        print(f"2. Paste the contents of: {schema_path}")
        print("3. Run the SQL to recreate tables, views, and functions")

    print(
        "\nTo clear matching and categorization tables only (non-destructive to content/taxonomy):"
    )
    print("- Use CLI match flags: --force-llm or --force-semantic for targeted cleanups")
    print("- Or run SQL to TRUNCATE matching_results, categorization_results (local-only)")


if __name__ == "__main__":
    main()


