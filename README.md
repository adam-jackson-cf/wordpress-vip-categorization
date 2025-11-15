# WordPress VIP Content Categorization

AI-powered content categorization system that ingests data from WordPress VIP API, uses OpenAI Batch API for categorization, and semantically matches content to taxonomy pages.

## Features

- **WordPress VIP Integration**: OSS connector for WordPress VIP API
- **Batch Processing**: Efficient OpenAI Batch API for large-scale categorization
- **Persistent Storage**: Supabase for data persistence across machines
- **Semantic Matching**: AI-powered matching of taxonomy to content pages
- **Prompt Optimization**: DSPy/Gepa integration for prompt engineering
- **Quality Gates**: Comprehensive testing and code quality checks
- **CSV Export**: Simple spreadsheet-based review workflow

## Architecture

This application uses a **cascading multi-stage workflow** for matching taxonomy pages to WordPress content:

1. **Stage 1: Semantic Matching** (threshold: 0.85)
   - Uses OpenAI-compatible embeddings to compute semantic similarity
   - Matches taxonomy pages to content based on category, description, and keywords
   - Items above 0.85 similarity are marked as matched

2. **Stage 2: LLM Categorization Fallback** (threshold: 0.9)
   - For items below semantic threshold, uses LLM to evaluate matches
   - LLM analyzes taxonomy page against all content to find best match
   - Items with confidence ≥ 0.9 are marked as matched

3. **Stage 3: Human Review**
   - Items failing both stages are marked for human review
   - Exported to CSV with blank target URLs for manual processing

Both stages can be enabled/disabled via configuration or CLI flags.

## Installation

### Prerequisites

- Python 3.10 or higher
- Supabase account and project
- API access for both semantic embeddings and LLM categorization (OpenRouter/OpenAI-compatible)
- WordPress VIP API access

### Setup

1. Clone the repository and navigate to the project:
```bash
cd wordpress-vip-categorization
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Copy the environment template and configure:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Initialize the database:
```bash
python -m src.cli init-db
```

## Configuration

Update `.env` with credentials for Supabase plus both AI providers (they can be the same service or separate):

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Semantic matching (embeddings)
SEMANTIC_API_KEY=sk-your-semantic-key
SEMANTIC_BASE_URL=https://openrouter.ai/api/v1
SEMANTIC_EMBEDDING_MODEL=qwen/qwen3-embedding-0.6b

# LLM categorization (chat completions)
LLM_API_KEY=sk-your-llm-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-exp:free
LLM_BATCH_TIMEOUT=86400

# WordPress ingest targets
WORDPRESS_VIP_SITES=https://wordpress.org/news
```

Legacy `OPENAI_*` variables are still read as fallbacks, but new deployments should use the `SEMANTIC_*` and `LLM_*` names to keep the two stages configurable.

## Usage

### 1. Prepare Taxonomy

Create a `taxonomy.csv` file with your source pages:
```csv
url,category,description
https://example.com/page1,Technology,Tech-related content
https://example.com/page2,Business,Business content
```

### 2. Ingest WordPress Content

```bash
python -m src.cli ingest --sites https://site1.com,https://site2.com
```

### 3. Run Cascading Matching Workflow

```bash
# Run full cascading workflow (semantic → LLM fallback)
python -m src.cli match

# Run with custom thresholds
python -m src.cli match --threshold 0.80

# Run semantic matching only (skip LLM stage)
python -m src.cli match --skip-llm

# Run LLM categorization only (skip semantic stage)
python -m src.cli match --skip-semantic

# Use non-batch mode for embeddings (slower, lower memory)
python -m src.cli match --no-batch
```

The `match` command runs the cascading workflow:
1. **Semantic matching** finds matches above 0.85 similarity (configurable)
2. **LLM categorization** evaluates remaining items (confidence ≥ 0.9)
3. **Unmatched items** are stored for human review

### 4. Export Results

```bash
python -m src.cli export --output results.csv
```

The exported CSV will contain:
- `source_url`: Taxonomy page URL
- `target_url`: Matched WordPress content URL (empty if no match)
- `category`: Category from taxonomy
- `similarity_score`: Semantic similarity score (0-1) or LLM confidence
- `confidence`: Categorization confidence (if categorized)
- `match_stage`: Stage where match was determined (`semantic_matched`, `llm_categorized`, or `needs_human_review`)
- `failed_at_stage`: Stage where matching failed (for debugging)

**Finding items for human review:** Filter for empty `target_url` or `match_stage=needs_human_review`.

## Troubleshooting

### Batch API taking too long
- Increase `LLM_BATCH_TIMEOUT` in `.env`
- Check batch status: `python -m src.cli batch-status --id <batch_id>`

### Supabase connection issues
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
- Ensure service role key is used (not anon key)

### Low matching quality
- Run DSPy optimization: `python -m src.cli optimize-prompts`
- Adjust similarity threshold in matching service
