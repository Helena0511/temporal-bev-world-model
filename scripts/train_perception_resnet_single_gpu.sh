#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_device="${RESWORLD_GPU:-4}"
if [[ -z "$gpu_device" || "$gpu_device" == *,* ]]; then
  echo 'RESWORLD_GPU must identify exactly one GPU' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_device"
python_bin="${RESWORLD_PYTHON:-python}"
run_path="${RESWORLD_OUTPUT:-$project_root/runs/city_phase1/perception_resnet50_ce_dice_v1}"
weights_path="$project_root/weights/resnet50-11ad3fa6.pth"
if [[ -e "$run_path" || -e "$run_path.log" ]]; then
  echo "Experiment already exists: $run_path" >&2
  exit 2
fi
if [[ ! -f "$weights_path" ]]; then
  echo "Missing verified ImageNet weights: $weights_path" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$run_path")"
"$python_bin" -c 'import os, torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print({"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "device": torch.cuda.get_device_name(0)})'
command=("$python_bin" -m scripts.train_staged_driving
  --stage perception --selection-file "$project_root/runs/city_phase1/full_selection.json"
  --model-config "$project_root/configs/bev_attention_city60_resnet50.json"
  --backbone-weights "$weights_path" --bev-loss ce_dice --backbone-lr-scale 0.1
  --image-height 256 --image-width 448 --batch-size 8 --epochs 4
  --workers 4 --threads 4 --device cuda --output "$run_path")
printf '%q ' "${command[@]}" > "$run_path.command.sh"
printf '\n' >> "$run_path.command.sh"
trap 'result=$?; printf "%s\n" "$result" > "$run_path.exit_code"' EXIT
cd -- "$project_root"
"${command[@]}" > "$run_path.log" 2>&1
