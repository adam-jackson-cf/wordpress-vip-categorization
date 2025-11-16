# Data & Supabase Rules

Applies to `src/data/` (Supabase client, schema helpers, exporters).

## Access Pattern
- All database interactions flow through `SupabaseClient`; never instantiate raw Supabase SDKs elsewhere.
- CRUD methods accept/return typed Pydantic models and hide JSON coercion from callers.

```python
class SupabaseClient:
    def __init__(self, settings: Settings) -> None:
        self._client = create_client(settings.supabase_url, settings.supabase_key)

    async def get_taxonomy(self, taxonomy_id: str) -> TaxonomyPage:
        record = await self._client.table("taxonomy_pages").select("*").eq("id", taxonomy_id).single()
        return TaxonomyPage.model_validate(record)
```

## Query Safety
- Use the query builder / RPC helpers for parameterized SQL; never concat user input into SQL strings.
- Multi-step writes use transactions when available; otherwise guard with explicit ordering and retries.

## Schema Ownership
- `src/data/schema.sql` is the source of truth; run `python -m src.cli init-db` after edits and keep schema + models in sync.
- Views like `export_results` should only be read via dedicated query helpers to avoid drift.

## Error Handling
- Wrap Supabase calls in try/except, log context (table, taxonomy id), and raise domain-specific errors upward.

```python
try:
    self._client.table("matching_results").insert(payload).execute()
except PostgrestError as exc:
    logger.error("Failed to insert match %s", payload["taxonomy_id"], exc_info=True)
    raise
```

## Checklist (`src/data/`)
- Only `SupabaseClient` touches the Supabase SDK; other modules depend on it instead of raw clients.
- All inputs/outputs are validated via Pydantic models and serializers.
- No string-concatenated SQL; everything uses parameterized builders or RPCs.
- Schema changes are mirrored in both `schema.sql` and any dependent models/tests immediately.
