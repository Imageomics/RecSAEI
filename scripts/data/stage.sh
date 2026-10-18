#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dataset="${DATASET:-nabirds}"
source_root="${SOURCE_ROOT:-$repo_root/data/$dataset}"
staged_root="${STAGED_ROOT:-$repo_root/data/staged/$dataset}"

cd "$repo_root"

cmd=(
  uv run --project "$repo_root" python -m "core.data.${dataset}_stage"
  "--${dataset}-root" "$source_root"
  --staged-root "$staged_root"
)
"${cmd[@]}"
