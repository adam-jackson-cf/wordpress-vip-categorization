# WordPress VIP Categorization - Setup Guide

## 📋 Prerequisites

1. **Supabase Account** - Sign up at [supabase.com](https://supabase.com)
2. **OpenAI Platform Access** - Create an org + API key at [platform.openai.com](https://platform.openai.com/) (the workflow uses `text-embedding-3-small` + `gpt-4o-mini`).
3. **Python 3.10+**

## 🚀 Quick Setup

### 1. Database Setup

#### Get Your Supabase Keys

1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. Copy your **URL** and **service_role key** (or anon key with RLS configured)

#### Run Database Schema

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Create a new query
4. Copy and paste the contents of `src/data/schema.sql`
5. Click **RUN** to execute

This will create the following tables:
- `wordpress_content` - Ingested WordPress posts/pages
- `taxonomy_pages` - Your source taxonomy for matching
- `categorization_results` - AI categorization results
- `matching_results` - Semantic matching between taxonomy and content
- `workflow_runs` - Metadata for resumable workflow executions

#### Enable pgvector & similarity helpers

The matching pipeline now stores embeddings directly in Supabase using [`pgvector`](https://supabase.com/docs/guides/database/extensions/pgvector). Make sure the extension is enabled after running the schema:

```sql
create extension if not exists vector;
```

The schema also creates `ivfflat` indexes plus a `match_wordpress_content` RPC to power fast `<->` lookups. After the first run, execute `ANALYZE wordpress_content;` so pgvector can tune the index.

### Bootstrap everything with one script

The repo ships with `scripts/bootstrap_supabase.py`, which sequentially calls the CLI (`init-db`, `load-taxonomy`, `ingest`, `match`) and optionally runs the recommended integration tests. Example:

```bash
scripts/bootstrap_supabase.py \
  --taxonomy-file data/Spain_New.csv \
  --sites https://wordpress.org/news \
  --max-pages 2 \
  --run-tests
```

Add `--include-slow-test` if you want it to run the full E2E pytest (sets `RUN_SLOW_TESTS=1`). You can skip individual stages via `--skip-...` flags if you simply need to re-run part of the pipeline.

### 2. Environment Configuration

Update `.env` with your credentials:

```bash
# Supabase - Use SERVICE_ROLE key for full access
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here

# Semantic Matching (Embeddings)
SEMANTIC_API_KEY=sk-openai-key
SEMANTIC_BASE_URL=https://api.openai.com        # the app appends /v1 automatically
SEMANTIC_EMBEDDING_MODEL=text-embedding-3-small
SEMANTIC_CANDIDATE_LIMIT=25

# LLM Categorization (OpenAI Batch)
LLM_API_KEY=sk-openai-key
LLM_BASE_URL=https://api.openai.com             # the app appends /v1 automatically
LLM_MODEL=gpt-4o-mini
LLM_BATCH_TIMEOUT=86400                         # workflow waits this long before timing out
LLM_BATCH_COMPLETION_WINDOW=24h                # window requested during batch submission
LLM_BATCH_CHUNK_SIZE=5000                      # requests per JSONL file before chunking
LLM_BATCH_ARTIFACT_DIR=data/batch              # where JSONL + manifest files land
LLM_CANDIDATE_LIMIT=10
LLM_CANDIDATE_MIN_SCORE=0.6

# DSPy prompt optimization (used by the batch fallback instructions)
DSPY_OPTIMIZATION_METRIC=accuracy

# Optional corporate CA bundle
ENABLE_CORP_CA=0
CORP_CA_BUNDLE_PATH=/path/to/corp-ca-bundle.pem

# WordPress sites to ingest (comma-separated `site|token`)
WORDPRESS_VIP_SITE_TOKENS="https://wordpress.org/news|example-token"

### Corporate CA helper

When `ENABLE_CORP_CA=1`, the Makefile automatically wraps Black/Mypy/Pytest through `scripts/corp_ca_exec.sh` so every tool honours `CORP_CA_BUNDLE_PATH`. Use the same helper for manual commands that need network access:

```bash
scripts/corp_ca_exec.sh python -m src.cli ingest --max-pages 1
```

# Batch tuning
INGESTION_BATCH_SIZE=200
MATCHING_BATCH_SIZE=200
```

### 3. Install Dependencies

```bash
cd wordpress-vip-categorization
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 4. Verify Setup

```bash
python scripts/test_setup.py
```

You should see:
```
✓ Configuration loaded successfully
✓ Supabase client initialized
✓ WordPress connection successful
✓ Generated embedding with 1024 dimensions
✓ ALL TESTS PASSED!
```

## 📊 Usage Workflow

### Step 1: Load Taxonomy

Create or update `data/Spain_New.csv`:
```csv
UID,Destination_URL,English_Page Name,Local_Page_Name,Content_Type,Primary_Audiance,Secondary_Audiance,Species,Semantic_Summary,Key_Topics,Reference_Source
TAX-001,https://example.com/wordpress,WordPress,WordPress,News,All,Media,WordPress news and updates,wordpress, cms, blogging
```

Then load it:
```bash
python -m src.cli load-taxonomy
```

### Step 2: Ingest WordPress Content

Ingest from configured sites:
```bash
python -m src.cli ingest
```

Or specify sites:
```bash
python -m src.cli ingest --sites https://site1.com,https://site2.com
```

Limit pages for testing:
```bash
python -m src.cli ingest --max-pages 2
```

Resume from the last successful pull or provide an explicit window:
```bash
python -m src.cli ingest --resume              # per-site checkpoint
python -m src.cli ingest --since 2025-11-01    # hard cutoff
```

### Step 3: Perform Semantic Matching

Match ingested content items to their best taxonomy destinations:
```bash
python -m src.cli match
```

With custom threshold:
```bash
python -m src.cli match --threshold 0.80
```

Targeted reruns:
```bash
# Retry just the backlog
python -m src.cli match --only-unmatched --skip-semantic --force-llm

# Focus on specific taxonomy rows (only those candidates will be considered)
python -m src.cli match --taxonomy-ids <uuid1,uuid2> --force-semantic

# Feed a CSV with a `url` column (used to whitelist destination URLs)
python -m src.cli match --taxonomy-file data/review_subset.csv
```

### Managing OpenAI Batch jobs directly

`match` waits for completion by default, but you can decouple the fallback stage for long-running datasets:

```bash
# Build candidate prompts for every needs_llm_review row and submit them asynchronously
python -m src.cli batch submit --limit 100 --no-wait

# Inspect progress for a given batch id
python -m src.cli batch status --id batch_abc123

# Once OpenAI marks the batch completed, apply the results to Supabase
python -m src.cli batch apply --id batch_abc123
```

All JSONL manifests live under `data/batch/<timestamp>/`. If you pass `--wait` to `batch submit` the CLI blocks until OpenAI finishes and immediately persists the parsed rubric decisions.

> **Note**: The Batch prompt is generated from the latest DSPy/GEPA-optimized matcher (instructions and demonstrations). Run `python -m src.cli optimize-dataset ...` followed by `scripts/promote_optimized_model.py` to refresh the production prompt without touching runtime code.

#### Managed workflow runs & resuming

Long-running cascades can be tracked and resumed via the `workflow` subcommands:

```bash
# Start with an explicit run key (defaults to run-<uuid>)
python -m src.cli workflow start --run-key nightly-2025-11-15

# Resume after fixing credentials or rate limits
python -m src.cli workflow resume nightly-2025-11-15

# Inspect history
python -m src.cli workflow status --limit 5
```

Each run updates the `workflow_runs` table with stage transitions, counts, and errors so you can continue from the last checkpoint.

### Step 4: Export Results

Export to CSV for review:
```bash
python -m src.cli export --output results/results.csv
```

The CSV will contain:
- `source_url` - WordPress content URL (the page being redirected)
- `target_url` - Taxonomy destination that content should redirect to (blank if still unresolved)
- `category` - Taxonomy content type/name
- `semantic_similarity_score` - Semantic match confidence (0-1)
- `match_stage` / `failed_at_stage` - Whether the row was accepted semantically, escalated to the LLM, or still requires human review

Filter for `target_url == ''` or `match_stage == needs_human_review` to find unmatched content needing analyst attention.

## 💰 Cost Management

The workflow calls two OpenAI models:

- **Embeddings** – `text-embedding-3-small` (~$0.00002 / 1K tokens)
- **LLM batch fallback** – OpenAI Batch pricing for `gpt-4o-mini` (50% off list: ~$0.0003 / 1K input tokens, ~$0.0012 / 1K output tokens)

Typical single-site smoke runs (≤10 content items) cost well under $0.05. Full runs scale linearly with the amount of content that needs either embeddings or LLM review.

**Cost-saving tips:**
1. Keep `--limit` low while iterating; only run the full dataset once semantic coverage looks healthy.
2. Set `ENABLE_LLM_CATEGORIZATION=0` when you only need semantic diagnostics.
3. Use `scripts/evaluate_semantic_thresholds.py` before launching LLM reruns—raising the semantic floor reduces downstream LLM spend.

## 🔍 Monitoring

### Check ingestion stats:
```bash
python -m src.cli stats
```

### Evaluate matching quality:
```bash
python -m src.cli evaluate
```

## 🐛 Troubleshooting

### Supabase Authentication Error

**Error:** `Invalid API key`

**Solutions:**
1. Use service_role key instead of anon key
2. Or configure Row Level Security policies for anon key:
   ```sql
   -- Enable RLS
   ALTER TABLE wordpress_content ENABLE ROW LEVEL SECURITY;
   ALTER TABLE taxonomy_pages ENABLE ROW LEVEL SECURITY;
   ALTER TABLE categorization_results ENABLE ROW LEVEL SECURITY;
   ALTER TABLE matching_results ENABLE ROW LEVEL SECURITY;

   -- Create policies for anon role
   CREATE POLICY "Enable all for anon" ON wordpress_content FOR ALL USING (true);
   CREATE POLICY "Enable all for anon" ON taxonomy_pages FOR ALL USING (true);
   CREATE POLICY "Enable all for anon" ON categorization_results FOR ALL USING (true);
   CREATE POLICY "Enable all for anon" ON matching_results FOR ALL USING (true);
   ```

### OpenAI Rate Limits

If you hit HTTP 429 or rate throttling:
1. Add short sleeps (`time.sleep(1-2)`) between retries during ingestion/matching.
2. Lower `--limit`, `SEMANTIC_CANDIDATE_LIMIT`, or `LLM_CANDIDATE_LIMIT` while iterating to keep request bursts small.
3. Consider enabling auto top-ups or purchasing additional OpenAI credit if you expect long-running full matches.

### WordPress API Issues

If WordPress fetches return 0 posts:
1. Check if the site has the REST API enabled
2. Try a different page number: `--max-pages 5`
3. Verify the site URL is correct

## 📁 Project Structure

```
wordpress-vip-categorization/
├── src/
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration
│   ├── models.py                 # Data models
│   ├── connectors/
│   │   └── wordpress_vip.py      # WordPress API
│   ├── data/
│   │   └── supabase_client.py    # Database
│   ├── services/
│   │   ├── ingestion.py          # Content ingestion
│   │   ├── matching.py           # Semantic matching ✅
│   │   └── categorization.py     # OpenAI Batch-powered fallback + rubric gating
│   └── exporters/
│       └── csv_exporter.py       # Results export
├── data/
│   └── Spain_New.csv             # Your taxonomy
├── src/
│   └── data/
│       └── schema.sql            # Database schema
├── scripts/
│   └── test_setup.py             # Setup verification
└── .env                          # Your credentials
```

## ✅ Tested Features

- ✅ WordPress VIP API connector
- ✅ Supabase persistence
- ✅ OpenAI embeddings (`text-embedding-3-small`)
- ✅ Semantic matching (content → taxonomy, cosine similarity)
- ✅ CSV export + analytics helpers
- ✅ OpenAI Batch-powered LLM fallback (rubric gated `gpt-4o-mini`)

## 🎯 Recommended Workflow

1. **Load your taxonomy** with up-to-date destinations and summaries.
2. **Ingest WordPress content** (start small with `--max-pages`/`--limit`).
3. **Run semantic matching** at the agreed threshold (≥0.70 for Spain migrations) and inspect `semantic_similarity_score`.
4. **Export and review** rows stuck in `needs_llm_review` / `needs_human_review`; use `results/semantic_miss_samples.csv` for diagnostics.
5. **Iterate** – adjust taxonomy metadata, content filters, or embeddings text, then rerun the workflow before launching the full tmux-managed match.

Following this loop keeps semantic coverage high (target ≥85%) and minimizes expensive LLM fallbacks.
