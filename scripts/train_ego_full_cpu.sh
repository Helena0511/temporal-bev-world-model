#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${RESWORLD_PYTHON:-python}"
run_path="$project_root/runs/city_phase1/ego_full_cpu"
if [[ -e "$run_path" || -e "$run_path.log" ]]; then
  echo "Experiment already exists: $run_path" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
command=("$python_bin" -m scripts.train_staged_driving
  --stage ego --selection-file "$project_root/runs/city_phase1/full_selection.json"
  --model-config "$project_root/configs/bev_attention_city60_pilot.json"
  --image-height 128 --image-width 352 --batch-size 64 --epochs 4
  --workers 4 --threads 4 --device cpu --output "$run_path")
printf '%q ' "${command[@]}" > "$run_path.command.sh"
printf '\n' >> "$run_path.command.sh"
trap 'result=$?; printf "%s\n" "$result" > "$run_path.exit_code"' EXIT
cd -- "$project_root"
"${command[@]}" > "$run_path.log" 2>&1
