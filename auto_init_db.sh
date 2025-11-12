#!/bin/bash
# Automated database initialization using Supabase API

set -e

echo "======================================================================="
echo "Initializing Supabase Database"
echo "======================================================================="

# Read environment variables
source .env

echo "Project: ${SUPABASE_URL}"

# Create SQL payload
SQL_FILE="schema.sql"

if [ ! -f "$SQL_FILE" ]; then
    echo "Error: schema.sql not found"
    exit 1
fi

# Read SQL content
SQL_CONTENT=$(<"$SQL_FILE")

echo ""
echo "Attempting to execute SQL via Supabase REST API..."
echo ""

# Try using curl to post to Supabase
# Note: This requires the Management API or a custom RPC function

# Extract project ref
PROJECT_REF=$(echo "$SUPABASE_URL" | sed 's/https:\/\///' | sed 's/\.supabase\.co.*//')

echo "Project reference: $PROJECT_REF"

# Try the query endpoint (may not work for DDL)
curl -X POST \
  "${SUPABASE_URL}/rest/v1/rpc/exec_sql" \
  -H "apikey: ${SUPABASE_KEY}" \
  -H "Authorization: Bearer ${SUPABASE_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\": $(cat schema.sql | jq -Rs .)}" \
  2>&1

echo ""
echo "======================================================================="
echo "If the above failed, please run schema.sql manually:"
echo "1. Go to: https://supabase.com/dashboard/project/${PROJECT_REF}"
echo "2. Click 'SQL Editor'"
echo "3. Click 'New Query'"
echo "4. Copy and paste the contents of schema.sql"
echo "5. Click 'RUN'"
echo "======================================================================="
