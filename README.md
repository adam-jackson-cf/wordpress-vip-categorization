# WordPress VIP Content Categorization

AI-powered content categorization system that ingests data from WordPress VIP API, uses OpenAI Batch API for categorization, and semantically matches content to taxonomy pages.

## Architecture

This application:
1. Ingests content from WordPress VIP sites using REST API
2. Stores data persistently in Supabase (works across different local machines)
3. Uses OpenAI Batch API for cost-effective LLM categorization
4. Implements semantic matching between source taxonomy pages and target content
5. Uses DSPy for prompt optimization and evaluation
6. Exports results as CSV for manual review

## Features

- **WordPress VIP Integration**: OSS connector for WordPress VIP API
- **Batch Processing**: Efficient OpenAI Batch API for large-scale categorization
- **Persistent Storage**: Supabase for data persistence across machines
- **Semantic Matching**: AI-powered matching of taxonomy to content pages
- **Prompt Optimization**: DSPy/Gepa integration for prompt engineering
- **Quality Gates**: Comprehensive testing and code quality checks
- **CSV Export**: Simple spreadsheet-based review workflow

## Installation

### Prerequisites

- Python 3.10 or higher
- Supabase account and project
- OpenAI API key
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

Create a `.env` file with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BATCH_TIMEOUT=86400  # 24 hours

# WordPress VIP Configuration
WORDPRESS_VIP_SITES=["https://site1.com", "https://site2.com"]
WORDPRESS_VIP_AUTH_TOKEN=your-auth-token  # Optional

# Application Configuration
TAXONOMY_FILE_PATH=./data/taxonomy.csv
BATCH_SIZE=1000
LOG_LEVEL=INFO
```

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

### 3. Run Categorization

```bash
python -m src.cli categorize --batch
```

### 4. Perform Semantic Matching

```bash
python -m src.cli match
```

### 5. Export Results

```bash
python -m src.cli export --output results.csv
```

The exported CSV will contain:
- `source_url`: Taxonomy page URL
- `target_url`: Matched WordPress content URL (empty if no match)
- `confidence`: Match confidence score
- `category`: Assigned category

Filter for empty `target_url` to find unmatched taxonomy pages.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_categorization.py

# Run integration tests only
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff src tests

# Type checking
mypy src
```

### Quality Gates

The project enforces:
- ✅ 80%+ test coverage
- ✅ Type hints on all functions
- ✅ Linting with ruff
- ✅ Code formatting with black
- ✅ All tests passing

## Project Structure

```
wordpress-vip-categorization/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration management
│   ├── models.py                 # Pydantic data models
│   ├── data/
│   │   ├── __init__.py
│   │   └── supabase_client.py   # Supabase integration
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── wordpress_vip.py     # WordPress VIP API connector
│   ├── services/
│   │   ├── __init__.py
│   │   ├── categorization.py    # OpenAI Batch categorization
│   │   ├── matching.py          # Semantic matching
│   │   └── ingestion.py         # Content ingestion orchestration
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── dspy_optimizer.py    # DSPy prompt optimization
│   │   └── evaluator.py         # Evaluation framework
│   └── exporters/
│       ├── __init__.py
│       └── csv_exporter.py      # CSV export functionality
├── tests/
│   ├── unit/
│   │   ├── test_wordpress_vip.py
│   │   ├── test_categorization.py
│   │   ├── test_matching.py
│   │   └── test_supabase_client.py
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── test_batch_processing.py
│   └── conftest.py              # Pytest fixtures
├── data/
│   └── taxonomy.csv.example     # Example taxonomy file
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── .env.example                 # Environment variables template
```

## Database Schema

The Supabase database uses the following tables:

### `wordpress_content`
- `id` (uuid, primary key)
- `url` (text, unique)
- `title` (text)
- `content` (text)
- `site_url` (text)
- `published_date` (timestamp)
- `metadata` (jsonb)
- `created_at` (timestamp)

### `taxonomy_pages`
- `id` (uuid, primary key)
- `url` (text, unique)
- `category` (text)
- `description` (text)
- `created_at` (timestamp)

### `categorization_results`
- `id` (uuid, primary key)
- `content_id` (uuid, foreign key)
- `category` (text)
- `confidence` (float)
- `batch_id` (text)
- `created_at` (timestamp)

### `matching_results`
- `id` (uuid, primary key)
- `taxonomy_id` (uuid, foreign key)
- `content_id` (uuid, foreign key)
- `similarity_score` (float)
- `created_at` (timestamp)

## Troubleshooting

### Batch API taking too long
- Increase `OPENAI_BATCH_TIMEOUT` in `.env`
- Check batch status: `python -m src.cli batch-status --id <batch_id>`

### Supabase connection issues
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
- Ensure service role key is used (not anon key)

### Low matching quality
- Run DSPy optimization: `python -m src.cli optimize-prompts`
- Adjust similarity threshold in matching service

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all quality gates pass
5. Submit a pull request
