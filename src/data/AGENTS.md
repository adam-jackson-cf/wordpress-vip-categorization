# Database & Supabase Development Standards

Project overview: [../../AGENTS.md](../../AGENTS.md)
Code review checklist: [../../CODE_REVIEW.md](../../CODE_REVIEW.md)

## SupabaseClient Wrapper Usage

```python
from src.data.supabase_client import SupabaseClient
from src.models import TaxonomyPage

client = SupabaseClient(settings)
taxonomy = client.get_taxonomy_by_id(taxonomy_id)  # Returns TaxonomyPage model
```

Never directly instantiate Supabase client - use `SupabaseClient` wrapper
Use type-safe models for all CRUD operations
Handle errors gracefully - wrap database calls in try-catch

## SQL Injection Prevention

Always use parameterized queries (Supabase client handles this)
Never concatenate user input into SQL queries
Validate all input before database operations

## Database Best Practices

- **Type Safety**: All CRUD methods return/accept Pydantic models
- **Error Handling**: Wrap all database operations in try-catch blocks
- **Transaction Safety**: Use transactions for multi-step operations
- **Query Efficiency**: Avoid N+1 queries; use joins and batch operations
- **Connection Management**: Let SupabaseClient manage connections

## Schema Reference

See `src/data/schema.sql` for complete schema definitions:
- `wordpress_content` - Ingested posts/pages
- `taxonomy_pages` - Source taxonomy
- `categorization_results` - AI categorization metadata
- `matching_results` - Match scores + stages
- `export_results` (view) - Joined projection for CSV export

## Common Pitfalls

1. **Direct Supabase Client**: Always use `SupabaseClient` wrapper
2. **SQL Injection**: Never concatenate user input into SQL queries
3. **Missing Error Handling**: Always wrap database calls in try-catch
4. **N+1 Queries**: Use efficient joins and batch operations

## Database Checklist (src/data)

- [ ] All database access goes through `SupabaseClient` (no direct client usage)
- [ ] Queries are parameterized (no string concatenation with user input)
- [ ] CRUD methods use typed Pydantic models for inputs/outputs
- [ ] Database calls are wrapped with error handling and appropriate logging
- [ ] Multi-step operations use transactions where consistency matters
- [ ] Queries are designed to avoid N+1 patterns (use joins/batching where appropriate)
