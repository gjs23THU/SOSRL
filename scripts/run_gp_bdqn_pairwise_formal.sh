#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PYTHON B2_CHECKPOINT G3_POLICY SCENARIO_DIR OUTPUT_ROOT" >&2
  exit 2
fi

python_bin=$1
b2_checkpoint=$2
g3_policy=$3
scenario_dir=$4
output_root=$5

mkdir -p "$output_root/logs"
exec 9>"$output_root/.launch.lock"
if ! flock -n 9; then
  echo "pairwise study is already running: $output_root" >&2
  exit 3
fi

expected_b2=48429459df21bc24d8b4a69f8c75a0ab7043146e69c104add62c1bd331314e5b
expected_g3=7d44fb425d75e83450ae9d90683432b765bd6b2aab8f9e22b8bd662187c548e9

check_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for $path: $actual" >&2
    exit 4
  fi
}

check_sha256 "$b2_checkpoint" "$expected_b2"
check_sha256 "$g3_policy" "$expected_g3"

arms=(additive pairwise_joint pairwise_residual)
ranks=(0 8 8)
warmups=(0 0 5000)
seeds=(16 17 18)

printf 'running\n' >"$output_root/status.txt"
git rev-parse HEAD >"$output_root/code_commit.txt"
sha256sum \
  "$scenario_dir/train.json" \
  "$scenario_dir/validation.json" \
  "$scenario_dir/test_iid.json" \
  "$scenario_dir/test_ood.json" \
  >"$output_root/scenario_sha256.txt"

for arm_index in "${!arms[@]}"; do
  arm=${arms[$arm_index]}
  rank=${ranks[$arm_index]}
  warmup=${warmups[$arm_index]}
  printf 'running %s\n' "$arm" >"$output_root/status.txt"
  pids=()
  for seed in "${seeds[@]}"; do
    cell="$output_root/$arm/seed_$seed"
    log="$output_root/logs/${arm}_seed_${seed}.log"
    if [[ -f "$cell/run_manifest.json" ]] && "$python_bin" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("stages",{}).get("input_integrity",{}).get("status")=="complete" else 1)' \
      "$cell/run_manifest.json"; then
      continue
    fi
    resume_args=()
    if [[ -f "$cell/run_manifest.json" ]]; then
      resume_args=(--resume)
    fi
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$python_bin" -m sosrl finetune-branching-with-gp \
      --scheduler-checkpoint "$b2_checkpoint" \
      --gp-policy "$g3_policy" \
      --scenario-dir "$scenario_dir" \
      --output-dir "$cell" \
      --extra-env-steps 40000 \
      --checkpoint-interval-steps 5000 \
      --lr 0.0001 \
      --lr-end 0.00001 \
      --lr-decay 0.9975 \
      --epsilon-start 0.10 \
      --epsilon-end 0.02 \
      --epsilon-decay 0.995 \
      --pairwise-interaction-rank "$rank" \
      --pairwise-residual-warmup-steps "$warmup" \
      --seed "$seed" \
      --device cuda \
      --skip-historical-test \
      "${resume_args[@]}" >"$log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

printf 'training_complete\n' >"$output_root/status.txt"
