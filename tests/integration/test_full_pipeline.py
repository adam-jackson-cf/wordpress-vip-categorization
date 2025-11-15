"""Live integration tests hitting Supabase and WordPress."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.exporters.csv_exporter import CSVExporter
from src.services.ingestion import IngestionService
from src.services.matching import MatchingService

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    if os.getenv("SUPABASE_TESTING") != "1":
        pytest.skip("Set SUPABASE_TESTING=1 to run live Supabase integration tests.")
    return Settings()


@pytest.fixture(scope="module")
def supabase_client(live_settings: Settings) -> SupabaseClient:
    return SupabaseClient(live_settings)


def _clear_supabase_tables(client: SupabaseClient, cutoff: str | None = None) -> None:
    tables = (
        "matching_results",
        "categorization_results",
        "wordpress_content",
        "taxonomy_pages",
    )
    for table in tables:
        query = client.client.table(table).delete()
        if cutoff:
            query = query.gte("created_at", cutoff)
        else:
            query = query.neq("id", "00000000-0000-0000-0000-000000000000")
        query.execute()


@pytest.fixture
def supabase_test_context(supabase_client: SupabaseClient):
    _clear_supabase_tables(supabase_client)
    start_time = datetime.now(timezone.utc)
    try:
        yield {"client": supabase_client, "started_at": start_time}
    finally:
        _clear_supabase_tables(supabase_client, start_time.isoformat())


@pytest.fixture(scope="module")
def public_wordpress_site() -> str:
    return "https://wordpress.org/news"


def _write_taxonomy(tmp_path: Path) -> Path:
    taxonomy_csv = tmp_path / "live_taxonomy.csv"
    taxonomy_csv.write_text(
        "url,category,description,keywords\n"
        "https://example.com/wp-news,WordPress News,WordPress announcements,wordpress;release;update;announcement\n"
        "https://example.com/tech,Technology,Tech articles,technology;development;programming\n"
    )
    return taxonomy_csv


@pytest.mark.live_supabase
@pytest.mark.timeout(120)
def test_full_e2e_pipeline_live(
    live_settings: Settings,
    supabase_client: SupabaseClient,
    supabase_test_context: dict[str, object],
    public_wordpress_site: str,
    tmp_path: Path,
) -> None:
    ingestion_service = IngestionService(live_settings, supabase_client)
    content_ingested = ingestion_service.ingest_wordpress_sites(
        [public_wordpress_site], max_pages=1
    )
    assert content_ingested > 0

    taxonomy_csv = _write_taxonomy(tmp_path)
    loaded = ingestion_service.load_taxonomy_from_csv(taxonomy_csv)
    assert loaded == 2

    matching_service = MatchingService(live_settings, supabase_client)
    results = matching_service.match_all_taxonomy_batch(min_threshold=0.0, store_results=True)
    assert len(results) == 2

    exporter = CSVExporter(supabase_client)
    output_path = tmp_path / "export.csv"
    exported = exporter.export_to_csv(output_path, include_unmatched=True)
    assert exported == 2
    assert output_path.exists()
    content = output_path.read_text()
    assert "WordPress News" in content
    assert "Technology" in content
