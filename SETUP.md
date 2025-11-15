# WordPress VIP Categorization - Setup Guide

## 📋 Prerequisites

1. **Supabase Account** - Sign up at [supabase.com](https://supabase.com)
2. **OpenRouter Account** - Sign up at [openrouter.ai](https://openrouter.ai)
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
4. Copy and paste the contents of `schema.sql`
5. Click **RUN** to execute

This will create the following tables:
- `wordpress_content` - Ingested WordPress posts/pages
- `taxonomy_pages` - Your source taxonomy for matching
- `categorization_results` - AI categorization results
- `matching_results` - Semantic matching between taxonomy and content

### 2. Environment Configuration

Update `.env` with your credentials:

```bash
# Supabase - Use SERVICE_ROLE key for full access
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here

# Semantic Matching (Embeddings)
SEMANTIC_API_KEY=sk-semantic-key
SEMANTIC_BASE_URL=https://openrouter.ai/api/v1
SEMANTIC_EMBEDDING_MODEL=qwen/qwen3-embedding-0.6b

# LLM Categorization (Chat)
LLM_API_KEY=sk-llm-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-exp:free
LLM_BATCH_TIMEOUT=86400

# WordPress sites to ingest (comma-separated)
WORDPRESS_VIP_SITES=https://wordpress.org/news
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
python test_setup.py
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

Create or update `data/taxonomy.csv`:
```csv
url,category,description,keywords
https://example.com/wordpress,WordPress,WordPress news,wordpress;cms;blogging
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

### Step 3: Perform Semantic Matching

Match taxonomy pages to ingested content:
```bash
python -m src.cli match
```

With custom threshold:
```bash
python -m src.cli match --threshold 0.80
```

### Step 4: Export Results

Export to CSV for review:
```bash
python -m src.cli export --output results.csv
```

The CSV will contain:
- `source_url` - Taxonomy page URL
- `target_url` - Matched WordPress content URL (empty if no match)
- `category` - Category name
- `similarity_score` - Match confidence (0-1)
- `confidence` - Categorization confidence (if categorized)

Filter for empty `target_url` to find unmatched taxonomy pages.

## 💰 Cost Management

### OpenRouter Free Tier

**Free Models Used:**
- **Chat**: `google/gemini-2.0-flash-exp:free`
- **Embeddings**: `qwen/qwen3-embedding-0.6b` (low cost)

**Rate Limits:**
- Free models: 20 requests/minute
- Daily limit: 50 calls

**Cost Estimates:**
- Embeddings: ~$0.00002 per 1K tokens
- For 100 pages with 1000 tokens each: ~$0.02

### ⚠️ Important Note: Batch API Not Supported

OpenRouter does not support OpenAI's Batch API. The categorization feature that uses batch processing will not work with OpenRouter.

**Alternatives:**
1. Skip categorization and use matching only
2. Use direct API calls (modify `src/services/categorization.py`)
3. Use OpenAI API for batch categorization separately

For this setup, **we recommend focusing on semantic matching**, which works perfectly with OpenRouter.

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

### OpenRouter Rate Limits

If you hit rate limits:
1. Add delays between requests
2. Process in smaller batches
3. Consider adding credits to your OpenRouter account for higher limits

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
│   │   └── categorization.py     # Categorization (⚠️ needs OpenAI Batch API)
│   └── exporters/
│       └── csv_exporter.py       # Results export
├── data/
│   └── taxonomy.csv              # Your taxonomy
├── schema.sql                    # Database schema
├── test_setup.py                 # Setup verification
└── .env                          # Your credentials
```

## ✅ Tested Features

- ✅ WordPress VIP API connector
- ✅ Supabase persistence
- ✅ OpenRouter embeddings (1024-dimensional)
- ✅ Semantic matching (0.94 similarity achieved in tests)
- ✅ CSV export
- ⚠️ Batch categorization (requires OpenAI API)

## 🎯 Recommended Workflow

For optimal results with OpenRouter:

1. **Load your taxonomy** with relevant keywords
2. **Ingest WordPress content** (start small, 10-20 pages)
3. **Run semantic matching** with threshold 0.70-0.80
4. **Export and review results** in spreadsheet
5. **Iterate**: Adjust keywords and threshold based on results

The semantic matching alone provides excellent results for URL redirects!
