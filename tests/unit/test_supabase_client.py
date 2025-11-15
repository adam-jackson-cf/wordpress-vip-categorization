"""Supabase client helper tests."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.data.supabase_client import SupabaseClient
from src.models import MatchStage


@pytest.fixture
def supabase_client(mocker, mock_settings) -> SupabaseClient:
    """Return a SupabaseClient backed by a fully mocked SDK client."""

    mock_sdk = MagicMock()
    mocker.patch("src.data.supabase_client.create_client", return_value=mock_sdk)
    client = SupabaseClient(mock_settings)
    client.client = mock_sdk
    return client


def _build_table_chain(mocker, data):
    table = mocker.Mock()
    table.select.return_value = table
    table.eq.return_value = table
    table.order.return_value = table
    table.limit.return_value = table
    table.in_.return_value = table
    table.delete.return_value = table
    table.gte.return_value = table
    table.execute.return_value = SimpleNamespace(data=data)
    return table


def test_get_latest_published_date_parses_iso(supabase_client, mocker):
    table = _build_table_chain(mocker, data=[{"published_date": "2024-01-01T12:00:00+00:00"}])
    supabase_client.client.table.return_value = table

    result = supabase_client.get_latest_published_date("https://example.com")

    assert isinstance(result, datetime)
    assert result.year == 2024
    supabase_client.client.table.assert_called_with("wordpress_content")


def test_get_taxonomy_by_urls_filters_empty(supabase_client, mocker, sample_taxonomy_page):
    table = _build_table_chain(mocker, data=[sample_taxonomy_page.model_dump(mode="json")])
    supabase_client.client.table.return_value = table

    rows = supabase_client.get_taxonomy_by_urls([str(sample_taxonomy_page.url)])

    assert len(rows) == 1
    assert rows[0].url == sample_taxonomy_page.url
    table.in_.assert_called_once()


def test_clear_matching_results_filters_and_logs(supabase_client, mocker):
    delete_table = mocker.Mock()
    delete_table.in_.return_value = delete_table
    delete_table.delete.return_value = delete_table
    delete_table.execute.return_value = SimpleNamespace(data=[{"id": "1"}])
    supabase_client.client.table.return_value = delete_table

    taxonomy_id = uuid4()
    deleted = supabase_client.clear_matching_results(
        taxonomy_ids=[taxonomy_id],
        stages=[MatchStage.LLM_CATEGORIZED],
    )

    assert deleted == 1
    supabase_client.client.table.assert_called_with("matching_results")
    delete_table.in_.assert_any_call("taxonomy_id", [str(taxonomy_id)])
    delete_table.in_.assert_any_call("match_stage", [MatchStage.LLM_CATEGORIZED.value])


def test_insert_content_returns_model(supabase_client, mocker, sample_wordpress_content):
    table = mocker.Mock()
    table.insert.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[sample_wordpress_content.model_dump(mode="json")]
    )
    supabase_client.client.table.return_value = table

    inserted = supabase_client.insert_content(sample_wordpress_content)

    assert inserted.url == sample_wordpress_content.url


def test_get_content_by_url_handles_missing(supabase_client, mocker):
    table = mocker.Mock()
    table.select.return_value = table
    table.eq.return_value = table
    table.execute.return_value = SimpleNamespace(data=[])
    supabase_client.client.table.return_value = table

    fetched = supabase_client.get_content_by_url("https://example.com")

    assert fetched is None


def test_get_content_by_id_returns_model(
    supabase_client,
    mocker,
    sample_wordpress_content,
) -> None:
    table = mocker.Mock()
    table.select.return_value = table
    table.eq.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[sample_wordpress_content.model_dump(mode="json")]
    )
    supabase_client.client.table.return_value = table

    result = supabase_client.get_content_by_id(sample_wordpress_content.id)

    assert result.id == sample_wordpress_content.id


def test_get_all_content_returns_list(
    supabase_client,
    mocker,
    sample_wordpress_content,
) -> None:
    table = mocker.Mock()
    table.select.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[sample_wordpress_content.model_dump(mode="json")]
    )
    supabase_client.client.table.return_value = table

    rows = supabase_client.get_all_content()

    assert len(rows) == 1


def test_get_taxonomy_by_ids_uses_in_clause(
    supabase_client,
    mocker,
    sample_taxonomy_page,
) -> None:
    table = mocker.Mock()
    table.select.return_value = table
    table.in_.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[sample_taxonomy_page.model_dump(mode="json")]
    )
    supabase_client.client.table.return_value = table

    rows = supabase_client.get_taxonomy_by_ids([sample_taxonomy_page.id])

    assert rows[0].id == sample_taxonomy_page.id
    table.in_.assert_called_once()


def test_get_matchings_by_taxonomy(
    supabase_client,
    mocker,
    sample_taxonomy_page,
    sample_wordpress_content,
) -> None:
    table = mocker.Mock()
    table.select.return_value = table
    table.eq.return_value = table
    table.order.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[
            {
                "id": str(uuid4()),
                "taxonomy_id": str(sample_taxonomy_page.id),
                "content_id": str(sample_wordpress_content.id),
                "similarity_score": 0.9,
                "match_stage": MatchStage.SEMANTIC_MATCHED.value,
            }
        ]
    )
    supabase_client.client.table.return_value = table

    rows = supabase_client.get_matchings_by_taxonomy(sample_taxonomy_page.id)

    assert rows[0].taxonomy_id == sample_taxonomy_page.id


def test_bulk_insert_no_records_short_circuits(supabase_client, mocker):
    supabase_client.bulk_insert("table", [])
    supabase_client.client.table.assert_not_called()


def test_bulk_insert_executes(supabase_client, mocker):
    table = mocker.Mock()
    table.insert.return_value = table
    table.execute.return_value = SimpleNamespace(data=[])
    supabase_client.client.table.return_value = table

    supabase_client.bulk_insert("matching_results", [{}])

    table.insert.assert_called_once_with([{}])


def test_upsert_taxonomy_and_get_by_id(
    supabase_client,
    mocker,
    sample_taxonomy_page,
) -> None:
    table = mocker.Mock()
    table.upsert.return_value = table
    table.select.return_value = table
    table.eq.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[sample_taxonomy_page.model_dump(mode="json")]
    )
    supabase_client.client.table.return_value = table

    stored = supabase_client.upsert_taxonomy(sample_taxonomy_page)
    assert stored.category == sample_taxonomy_page.category

    fetched = supabase_client.get_taxonomy_by_id(sample_taxonomy_page.id)
    assert fetched.url == sample_taxonomy_page.url


def test_get_categorizations_by_content(
    supabase_client,
    mocker,
    sample_wordpress_content,
) -> None:
    table = mocker.Mock()
    table.select.return_value = table
    table.eq.return_value = table
    table.execute.return_value = SimpleNamespace(
        data=[
            {
                "id": str(uuid4()),
                "content_id": str(sample_wordpress_content.id),
                "category": "Tech",
                "confidence": 0.9,
                "created_at": datetime.utcnow().isoformat(),
            }
        ]
    )
    supabase_client.client.table.return_value = table

    rows = supabase_client.get_categorizations_by_content(sample_wordpress_content.id)
    assert rows[0].category == "Tech"


def test_get_best_match_for_taxonomy_returns_none(
    supabase_client,
    mocker,
    sample_taxonomy_page,
) -> None:
    table = mocker.Mock()
    table.select.return_value = table
    table.eq.return_value = table
    table.gte.return_value = table
    table.order.return_value = table
    table.limit.return_value = table
    table.execute.return_value = SimpleNamespace(data=[])
    supabase_client.client.table.return_value = table

    result = supabase_client.get_best_match_for_taxonomy(sample_taxonomy_page.id, min_score=0.5)
    assert result is None
