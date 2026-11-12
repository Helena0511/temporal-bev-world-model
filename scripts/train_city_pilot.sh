#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
selection_path="$project_root/runs/city_phase1/selection.json"
data_root="${RESWORLD_DATA_ROOT:-/mnt/ssd0/datasets/nuscenes}"
cd -- "$project_root"
if [[ ! -f "$selection_path" ]]; then
  python -m scripts.select_city_data --root "$data_root" --output "$selection_path"
fi
exec python -m scripts.train_driving \
  --root "$data_root" --version v1.0-trainval --mode train \
  --model-config "$project_root/configs/bev_attention_city60_pilot.json" \
  --selection-file "$selection_path" \
  --epochs 2 --batch-size 2 --image-height 64 --image-width 160 \
  --device "${RESWORLD_DEVICE:-cpu}" --threads 4 \
  --output "$project_root/runs/city_phase1/pilot_train" "$@"
