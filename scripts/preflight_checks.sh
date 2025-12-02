#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +o allexport
else
  echo "⚠️  .env file not found at $ENV_FILE; aborting." >&2
  exit 1
fi

: "${WORDPRESS_VIP_SITE_TOKENS:?WORDPRESS_VIP_SITE_TOKENS not set in .env}"
: "${LLM_API_KEY:?LLM_API_KEY not set in .env}"
: "${LLM_BASE_URL:?LLM_BASE_URL not set in .env}"

if [[ "${ENABLE_CORP_CA:-0}" == "1" ]]; then
  : "${CORP_CA_BUNDLE_PATH:?CORP_CA_BUNDLE_PATH must be set when ENABLE_CORP_CA=1}"
  export SSL_CERT_FILE="$CORP_CA_BUNDLE_PATH"
  export REQUESTS_CA_BUNDLE="$CORP_CA_BUNDLE_PATH"
  export UV_HTTP_CA_BUNDLE="$CORP_CA_BUNDLE_PATH"
  export PIP_CERT="$CORP_CA_BUNDLE_PATH"
fi

echo "== WordPress VIP endpoint checks =="
IFS=',' read -ra WP_ENTRIES <<< "$WORDPRESS_VIP_SITE_TOKENS"
wp_failures=0

for entry in "${WP_ENTRIES[@]}"; do
  pair="$(printf '%s' "$entry" | tr -d ' ')"
  site="${pair%%|*}"
  token="${pair#*|}"
  site="$(printf '%s' "$site" | sed 's#/*$##')"
  url="${site}/wp-json/?token=${token}"
  printf 'Checking %s ... ' "$site"
  status=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "$url" || echo "000")
  if [[ "$status" == "200" ]]; then
    echo "OK (HTTP 200)"
  else
    echo "FAIL (HTTP $status)"
    ((wp_failures++))
  fi
done

echo
echo "== OpenAI connectivity check =="
openai_endpoint="${LLM_BASE_URL%/}/v1/models"
status=$(curl -sS -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  -H "Content-Type: application/json" \
  --max-time 15 \
  "$openai_endpoint" || echo "000")

if [[ "$status" == "200" ]]; then
  echo "OpenAI model listing reachable (HTTP 200)"
else
  echo "OpenAI connectivity failed (HTTP $status)"
fi

echo
if [[ "$wp_failures" -eq 0 && "$status" == "200" ]]; then
  echo "✅ Preflight checks passed"
else
  echo "❌ Preflight checks reported issues"
  exit 1
fi
