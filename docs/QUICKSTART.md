# 🚀 Quick Start Guide

## Step 1: Initialize Database (1 minute)

Your Supabase project is ready, we just need to create the tables.

### Execute the Schema:

1. Open `src/data/schema.sql` and copy its contents. That file is the single source of truth for all tables/indexes/views.
2. **Open your Supabase project:** 👉 https://supabase.com/dashboard/project/resciqkhyvnaxqpzabtb
3. Click **"SQL Editor"** → **"New Query"**, paste the contents of `src/data/schema.sql`, and run it (Ctrl/Cmd + Enter).
4. Wait for the success message (usually ~2 seconds) and you’re ready for automated setup.

## Step 2: Run Automated Setup

Once the tables are created, run:

```bash
cd wordpress-vip-categorization
source venv/bin/activate
python -m src.cli full-run --output results/results.csv
```

This will automatically:
- ✅ Load your taxonomy (4 sample pages)
- ✅ Ingest WordPress content (~10-20 posts from WordPress.org)
- ✅ Run semantic matching with OpenRouter embeddings
- ✅ Export a single `results/results.csv` where unmatched rows simply have a blank `target_url`

**Expected runtime:** 2-3 minutes
**Expected cost:** < $0.01 (using free/low-cost models)

## Step 3: Review Results

Open `results/results.csv` in your spreadsheet application:

- **source_url**: Your taxonomy page
- **target_url**: Matched WordPress content (empty if no match)
- **similarity_score**: Match confidence (0.0-1.0)
- **category**: Category name

**To find unmatched items:** Filter for empty `target_url` (or `match_stage == needs_human_review`).

## 📎 Next Steps

- Need to customize ingestion, adjust thresholds, or troubleshoot? See the detailed [SETUP guide](SETUP.md).
- Want to understand the full architecture and developer workflow? Read [AGENTS.md](../AGENTS.md).

🎉 Once the schema is in place, `python -m src.cli full-run --output results/results.csv` gives you export-ready matches in minutes.
