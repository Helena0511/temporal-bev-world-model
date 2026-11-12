#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
selection_path="${RESWORLD_SELECTION_FILE:-$project_root/runs/city_phase1/full_selection.json}"
cd -- "$project_root"
if [[ ! -f "$selection_path" ]]; then
  python -m scripts.select_city_data \
    --root "${RESWORLD_DATA_ROOT:-/mnt/ssd0/datasets/nuscenes}" \
    --all-eligible --output "$selection_path"
fi
exec python -m scripts.train_driving \
  --root "${RESWORLD_DATA_ROOT:-/mnt/ssd0/datasets/nuscenes}" \
  --version v1.0-trainval \
  --mode train \
  --selection-file "$selection_path" \
  --model-config "${RESWORLD_MODEL_CONFIG:-$project_root/configs/bev_attention_city60.json}" \
  --image-height 128 --image-width 352 \
  --batch-size 2 --epochs 12 --lr 0.0003 \
  --workers 4 --threads 4 \
  --device "${RESWORLD_DEVICE:-cuda}" \
  --output "$project_root/runs/trainval_native" "$@"
