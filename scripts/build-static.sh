#!/usr/bin/env bash
# Build the static export for GitHub Pages (or any static host).
# API route handlers cannot be included in a static export, so they are
# temporarily moved aside for the build and restored afterwards.
#
# Usage:  bash scripts/build-static.sh [BASE_PATH]
#   BASE_PATH defaults to /AI-News (repo is hosted at https://<user>.github.io/AI-News)
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_PATH="${1:-/AI-News}"
echo "→ Building static export (basePath=${BASE_PATH})"

# API route handlers cannot be included in a static export. Move them OUTSIDE
# the app/ directory (any folder under app/ becomes a route) and restore after.
if [ -d app/api ]; then
  echo "→ Moving app/api out of the build tree"
  mkdir -p .api-disabled
  mv app/api .api-disabled/api
fi

cleanup() {
  if [ -d .api-disabled/api ]; then
    mv .api-disabled/api app/api
    rmdir .api-disabled 2>/dev/null || true
    echo "→ Restored app/api"
  fi
}
trap cleanup EXIT

NEXT_PUBLIC_STATIC=true NEXT_PUBLIC_BASE_PATH="${BASE_PATH}" npx next build

cleanup
trap - EXIT
echo "→ Static export written to out/"
ls out/ | head
