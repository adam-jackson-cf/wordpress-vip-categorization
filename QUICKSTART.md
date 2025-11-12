# 🚀 Quick Start Guide

## Step 1: Initialize Database (1 minute)

Your Supabase project is ready, we just need to create the tables.

### Copy This SQL:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- WordPress content table
CREATE TABLE IF NOT EXISTS wordpress_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    site_url TEXT NOT NULL,
    published_date TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wordpress_content_url ON wordpress_content(url);
CREATE INDEX IF NOT EXISTS idx_wordpress_content_site_url ON wordpress_content(site_url);

-- Taxonomy pages table
CREATE TABLE IF NOT EXISTS taxonomy_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_url ON taxonomy_pages(url);
CREATE INDEX IF NOT EXISTS idx_taxonomy_pages_category ON taxonomy_pages(category);

-- Categorization results table
CREATE TABLE IF NOT EXISTS categorization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES wordpress_content(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    batch_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_categorization_content_id ON categorization_results(content_id);
CREATE INDEX IF NOT EXISTS idx_categorization_batch_id ON categorization_results(batch_id);

-- Matching results table
CREATE TABLE IF NOT EXISTS matching_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    taxonomy_id UUID NOT NULL REFERENCES taxonomy_pages(id) ON DELETE CASCADE,
    content_id UUID REFERENCES wordpress_content(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(taxonomy_id, content_id)
);

CREATE INDEX IF NOT EXISTS idx_matching_taxonomy_id ON matching_results(taxonomy_id);
CREATE INDEX IF NOT EXISTS idx_matching_content_id ON matching_results(content_id);
CREATE INDEX IF NOT EXISTS idx_matching_similarity_score ON matching_results(similarity_score DESC);

-- View for easy export
CREATE OR REPLACE VIEW export_results AS
SELECT
    tp.url as source_url,
    COALESCE(wc.url, '') as target_url,
    COALESCE(mr.similarity_score, 0.0) as similarity_score,
    tp.category,
    COALESCE(cr.confidence, 0.0) as confidence
FROM taxonomy_pages tp
LEFT JOIN matching_results mr ON tp.id = mr.taxonomy_id
LEFT JOIN wordpress_content wc ON mr.content_id = wc.id
LEFT JOIN categorization_results cr ON wc.id = cr.content_id
ORDER BY tp.category, mr.similarity_score DESC NULLS LAST;
```

### Execute the SQL:

1. **Open your Supabase project:**
   👉 https://supabase.com/dashboard/project/resciqkhyvnaxqpzabtb

2. **Click "SQL Editor"** in the left sidebar

3. **Click "New Query"**

4. **Copy the SQL above and paste it**

5. **Click "RUN"** (or press Ctrl+Enter)

6. **Wait for "Success" message** (takes ~2 seconds)

## Step 2: Run Automated Setup

Once the tables are created, run:

```bash
cd wordpress-vip-categorization
source venv/bin/activate
python run_setup.py
```

This will automatically:
- ✅ Load your taxonomy (4 sample pages)
- ✅ Ingest WordPress content (~10-20 posts from WordPress.org)
- ✅ Run semantic matching with OpenRouter embeddings
- ✅ Export results to `results.csv` and `unmatched.csv`

**Expected runtime:** 2-3 minutes
**Expected cost:** < $0.01 (using free/low-cost models)

## Step 3: Review Results

Open `results.csv` in your spreadsheet application:

- **source_url**: Your taxonomy page
- **target_url**: Matched WordPress content (empty if no match)
- **similarity_score**: Match confidence (0.0-1.0)
- **category**: Category name

**To find unmatched items:** Filter for empty `target_url` or open `unmatched.csv`

## 🎯 What's Configured

- **Supabase**: Your database (service_role key configured)
- **OpenRouter**: Using free models
  - Chat: `google/gemini-2.0-flash-exp:free` (no cost)
  - Embeddings: `qwen/qwen3-embedding-0.6b` (~$0.00002/1K tokens)
- **WordPress Source**: wordpress.org/news (for testing)
- **Similarity Threshold**: 0.70 (70% match required)

## 🔧 Customization

### Use Your Own WordPress Sites

Edit `.env`:
```bash
WORDPRESS_VIP_SITES=https://yoursite1.com,https://yoursite2.com
```

### Use Your Own Taxonomy

Edit `data/taxonomy.csv`:
```csv
url,category,description,keywords
https://yoursite.com/page1,Category1,Description,keyword1;keyword2;keyword3
```

### Adjust Match Threshold

Edit `.env`:
```bash
SIMILARITY_THRESHOLD=0.80  # Require 80% match (stricter)
```

Then rerun:
```bash
python -m src.cli match --threshold 0.80
python -m src.cli export --output results.csv
```

## ❓ Troubleshooting

### "Tables missing" error
➡️ Run the SQL in Supabase SQL Editor (Step 1 above)

### "No posts fetched"
➡️ The site might not have /wp-json/wp/v2/ API enabled
➡️ Try a different site or check the WordPress REST API documentation

### "Rate limit" errors
➡️ OpenRouter free tier: 20 requests/minute
➡️ Add small delays or reduce batch size in `.env`

## 📊 Sample Output

```
Matching 4 taxonomy pages to 15 content items...
✓ Matched 3/4 taxonomy pages

Sample matches:
  1. https://example.com/wordpress
     → https://wordpress.org/news/2024/01/wordpress-announcement
     Score: 0.9234
```

## 🎉 You're All Set!

Once tables are created, everything else is automated.
Run `python run_setup.py` and you'll have your results in minutes!
