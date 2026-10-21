#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dataset="${DATASET:-nabirds}"
token_mode="${TOKEN_MODE:-content}" # content or cls
shards_root="${SHARDS_ROOT:-$repo_root/artifacts/saev/shards}"
seed="${SHUFFLE_SEED:-42}"
layer="${LAYER:--2}"
case "$token_mode" in
  content) ;;
  cls) echo "CLS training does not need patch-token shuffling."; exit 0 ;;
  *) echo "Unknown token_mode: $token_mode" >&2; exit 1 ;;
esac
train_tag="$dataset-raev2-dinov3-l-i256-train-l$layer-content-t256"
source_shards="${SOURCE_SHARDS:-$shards_root/$train_tag}"
output_tag="${SHUFFLED_TRAIN_TAG:-$train_tag-shuffled-s$seed}"

cd "$repo_root"
cmd=(
  uv run --project "$repo_root" python -m core.data.token_shuffle
  --source-shards "$source_shards" --shards-root "$shards_root"
  --output-tag "$output_tag" --seed "$seed"
  --source-chunk-examples "${SHUFFLE_SOURCE_CHUNK_EXAMPLES:-128}"
  --write-chunk-tokens "${SHUFFLE_WRITE_CHUNK_TOKENS:-8192}"
)
if [[ -n "${SHUFFLE_MAX_TOKENS:-}" ]]; then
  cmd+=(--max-tokens "$SHUFFLE_MAX_TOKENS")
fi
"${cmd[@]}"
