#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CORP_CA_BUNDLE_PATH:-}" ]]; then
  echo "ERROR: CORP_CA_BUNDLE_PATH is not set. Update .env before enabling corp CA." >&2
  exit 1
fi

if [[ ! -f "$CORP_CA_BUNDLE_PATH" ]]; then
  echo "ERROR: CA bundle not found at $CORP_CA_BUNDLE_PATH" >&2
  exit 1
fi

export SSL_CERT_FILE="$CORP_CA_BUNDLE_PATH"
export REQUESTS_CA_BUNDLE="$CORP_CA_BUNDLE_PATH"
export UV_HTTP_CA_BUNDLE="$CORP_CA_BUNDLE_PATH"
export PIP_CERT="$CORP_CA_BUNDLE_PATH"

exec "$@"
