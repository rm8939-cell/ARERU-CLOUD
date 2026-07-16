#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="${HOME}/.local/bin:${PATH}"
export PORT="${PORT:-5001}"

echo "ARERU.CLOUD β starting on :${PORT}"
exec gunicorn web_app:app --bind "0.0.0.0:${PORT}" --timeout 120 --workers 2
