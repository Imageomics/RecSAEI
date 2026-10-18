#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Edit the settings below before running; use matching values across stages.
dataset="${DATASET:-nabirds}"
token_mode="${TOKEN_MODE:-content}" # content or cls
shards_root="${SHARDS_ROOT:-$repo_root/artifacts/saev/shards}"
train_root="${STAGED_ROOT:-$repo_root/data/staged/$dataset}/train"
val_root="${STAGED_ROOT:-$repo_root/data/staged/$dataset}/val"
ckpt="${CKPT:-$repo_root/RAEv2/pretrained_models/encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
d_model=1024
content_tokens_per_example=256
extraction_batch_size="${EXTRACTION_BATCH_SIZE:-128}"
n_workers="${N_WORKERS:-8}"
max_tokens_per_shard=1800000

case "$token_mode" in
  content)
    layer=-2
    cls_args=(--no-cls-token)
    ;;
  cls)
    layer=-1
    cls_args=(--cls-token)
    ;;
  *) echo "Unknown token_mode: $token_mode" >&2; exit 1 ;;
esac

layer="${LAYER:-$layer}"
prefix="$dataset-raev2-dinov3-l-i256"
train_tag="$prefix-train-l$layer-$token_mode-t256"
val_tag="$prefix-val-l$layer-$token_mode-t256"

cd "$repo_root"
mkdir -p "$shards_root"
for tag in "$train_tag" "$val_tag"; do
  if [[ -e "$shards_root/$tag" ]]; then
    echo "Shards already exist: $shards_root/$tag" >&2
    exit 1
  fi
done

for split in train val; do
  tag="$train_tag"
  data_root="$train_root"
  if [[ "$split" == val ]]; then tag="$val_tag"; data_root="$val_root"; fi
  cmd=(
    uv run --project "$repo_root" python "$repo_root/saev/launch.py" shards
    --shards-root "$shards_root"
    --family dinov3 --ckpt "$ckpt"
    --tag "$tag"
    --d-model "$d_model"
    --layers "$layer"
    --content-tokens-per-example "$content_tokens_per_example"
    "${cls_args[@]}"
    --image-resize-mode square
    --batch-size "$extraction_batch_size"
    --n-workers "$n_workers"
    --max-tokens-per-shard "$max_tokens_per_shard"
    --device cuda
    data:img-folder --data.root "$data_root"
  )
  "${cmd[@]}"
done
