#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 PYTHON ROUND1_SCENARIO_ROOT PREVIOUS_EVAL_ROOT BASE_G2_POLICY BASE_B2_CHECKPOINT OUTPUT_ROOT EXPECTED_COMMIT" >&2
  exit 2
fi

PYTHON_BIN=$1
ROUND1_ROOT=$2
PREVIOUS_EVAL_ROOT=$3
BASE_GP_POLICY=$4
BASE_BDQN_CHECKPOINT=$5
OUTPUT_ROOT=$6
EXPECTED_COMMIT=$7
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCENARIO_ROOT="$OUTPUT_ROOT/scenarios"
RUN_OUTPUT="$OUTPUT_ROOT/continuation_120x50x3_40k"
STATUS_PATH="$OUTPUT_ROOT/pipeline_status.json"
MAX_CPU_LOAD=${SOSRL_MAX_CPU_LOAD:-16.0}
TEST_IID_MANIFEST="$ROUND1_ROOT/test_iid_v2.json"
TEST_OOD_MANIFEST="$ROUND1_ROOT/test_ood_v2.json"
if [[ ! -f "$TEST_IID_MANIFEST" ]]; then
  TEST_IID_MANIFEST="$ROUND1_ROOT/test_iid.json"
fi
if [[ ! -f "$TEST_OOD_MANIFEST" ]]; then
  TEST_OOD_MANIFEST="$ROUND1_ROOT/test_ood.json"
fi

if [[ -e "$OUTPUT_ROOT" && "${SOSRL_ALLOW_RESUME:-0}" != "1" ]]; then
  echo "output root already exists; set SOSRL_ALLOW_RESUME=1 to resume: $OUTPUT_ROOT" >&2
  exit 5
fi
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "another continuation pipeline holds $OUTPUT_ROOT/pipeline.lock" >&2
  exit 3
fi

EXISTING_MANIFESTS=(
  "$ROUND1_ROOT/b/train.json"
  "$ROUND1_ROOT/b/validation.json"
  "$ROUND1_ROOT/g/train.json"
  "$ROUND1_ROOT/g/validation.json"
  "$TEST_IID_MANIFEST"
  "$TEST_OOD_MANIFEST"
  "$PREVIOUS_EVAL_ROOT/gate_iid.json"
  "$PREVIOUS_EVAL_ROOT/gate_ood.json"
  "$PREVIOUS_EVAL_ROOT/final_iid.json"
  "$PREVIOUS_EVAL_ROOT/final_ood.json"
)
for required in "$PYTHON_BIN" "$BASE_GP_POLICY" "$BASE_BDQN_CHECKPOINT" "${EXISTING_MANIFESTS[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 4
  fi
done

update_status() {
  local stage=$1
  local state=$2
  local details=${3:-}
  "$PYTHON_BIN" -c 'import datetime,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); payload={"stage":sys.argv[2],"status":sys.argv[3],"details":sys.argv[4],"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_commit":sys.argv[5],"pid":int(sys.argv[6])}; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); t.replace(p)' \
    "$STATUS_PATH" "$stage" "$state" "$details" "$EXPECTED_COMMIT" "$$"
}

CURRENT_STAGE=preflight
record_failure() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    update_status "$CURRENT_STAGE" failed "exit_code=$exit_code" || true
  fi
  return "$exit_code"
}
trap record_failure EXIT

update_status preflight running
ACTUAL_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "code SHA mismatch: expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 6
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "training worktree is not clean: $REPO_ROOT" >&2
  exit 7
fi
BASE_GP_SHA=$(sha256sum "$BASE_GP_POLICY" | awk '{print $1}')
BASE_BDQN_SHA=$(sha256sum "$BASE_BDQN_CHECKPOINT" | awk '{print $1}')
if [[ -n "${SOSRL_EXPECTED_G2_SHA:-}" && "$BASE_GP_SHA" != "$SOSRL_EXPECTED_G2_SHA" ]]; then
  echo "G2 SHA mismatch: expected $SOSRL_EXPECTED_G2_SHA, got $BASE_GP_SHA" >&2
  exit 8
fi
if [[ -n "${SOSRL_EXPECTED_B2_SHA:-}" && "$BASE_BDQN_SHA" != "$SOSRL_EXPECTED_B2_SHA" ]]; then
  echo "B2 SHA mismatch: expected $SOSRL_EXPECTED_B2_SHA, got $BASE_BDQN_SHA" >&2
  exit 9
fi
"$PYTHON_BIN" -c 'import torch,deap; assert torch.cuda.is_available(), "CUDA is required for BDQN"; print({"torch":torch.__version__,"cuda_device":torch.cuda.get_device_name(0),"deap":deap.__version__})'
sha256sum "$BASE_GP_POLICY" "$BASE_BDQN_CHECKPOINT" "${EXISTING_MANIFESTS[@]}" \
  | tee "$OUTPUT_ROOT/logs/input_sha256.txt"
update_status preflight complete "g2_sha=$BASE_GP_SHA b2_sha=$BASE_BDQN_SHA"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CURRENT_STAGE=tests
update_status tests running
"$PYTHON_BIN" -m pytest -q tests/test_alternating_stack.py tests/test_cli.py \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/tests.log"
update_status tests complete

SCENARIO_ARGS=()
for manifest in "${EXISTING_MANIFESTS[@]}"; do
  SCENARIO_ARGS+=(--existing-manifest "$manifest")
done

CURRENT_STAGE=scenario_generation
update_status scenario_generation running
if [[ ! -f "$SCENARIO_ROOT/scenario_registry.json" ]]; then
  "$PYTHON_BIN" -m sosrl generate-alternation-scenarios \
    "${SCENARIO_ARGS[@]}" \
    --gate-iid-seed 20261010 \
    --gate-ood-seed 20261011 \
    --final-iid-seed 20261012 \
    --final-ood-seed 20261013 \
    --output-dir "$SCENARIO_ROOT"
fi
sha256sum "$SCENARIO_ROOT"/*.json | tee "$OUTPUT_ROOT/logs/scenario_sha256.txt"
"$PYTHON_BIN" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); payload={"schema_version":1,"global_stage_mapping":{"S0":"S2","SG1":"SG3","S1":"S3","SG2":"SG4","S2":"S4"},"global_policy_mapping":{"G0":"G2","G1":"G3","G2":"G4","B0":"B2","B1":"B3","B2":"B4"},"base_inputs":{"gp_policy":sys.argv[2],"scheduler_checkpoint":sys.argv[3]}}; p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")' \
  "$OUTPUT_ROOT/lineage_mapping.json" "$BASE_GP_POLICY" "$BASE_BDQN_CHECKPOINT"
update_status scenario_generation complete

resource_snapshot() {
  local cpu_load mem_kib gpu_mib disk_kib
  cpu_load=$(awk '{print $1}' /proc/loadavg)
  mem_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  gpu_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -n1)
  disk_kib=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
  printf '%s %s %s %s\n' "$cpu_load" "$mem_kib" "$gpu_mib" "$disk_kib"
}

CURRENT_STAGE=resource_gate
while true; do
  read -r CPU_LOAD MEM_KIB GPU_MIB DISK_KIB < <(resource_snapshot)
  RESOURCE_DETAILS="cpu_load=$CPU_LOAD max_cpu_load=$MAX_CPU_LOAD mem_available_kib=$MEM_KIB gpu_free_mib=$GPU_MIB disk_free_kib=$DISK_KIB"
  if awk -v load="$CPU_LOAD" -v maximum="$MAX_CPU_LOAD" 'BEGIN {exit !(load <= maximum)}' \
    && (( MEM_KIB >= 16777216 )) \
    && (( GPU_MIB >= 24576 )) \
    && (( DISK_KIB >= 52428800 )); then
    update_status resource_gate complete "$RESOURCE_DETAILS"
    break
  fi
  update_status resource_gate waiting "$RESOURCE_DETAILS"
  echo "continuation waiting for resources: $RESOURCE_DETAILS"
  sleep 300
done

CURRENT_STAGE=continuation
update_status continuation running
SOSRL_ALLOW_RESUME=1 "$PYTHON_BIN" -u -m sosrl run-gp-bdqn-alternation \
  --base-gp-policy "$BASE_GP_POLICY" \
  --base-scheduler-checkpoint "$BASE_BDQN_CHECKPOINT" \
  --scenario-dir "$ROUND1_ROOT/g" \
  --gate-iid-manifest "$SCENARIO_ROOT/gate_iid.json" \
  --gate-ood-manifest "$SCENARIO_ROOT/gate_ood.json" \
  --final-iid-manifest "$SCENARIO_ROOT/final_iid.json" \
  --final-ood-manifest "$SCENARIO_ROOT/final_ood.json" \
  --output-dir "$RUN_OUTPUT" \
  --gp-population-size 120 \
  --gp-max-generations 50 \
  --gp-runs 3 \
  --gp-workers 12 \
  --gp-min-generations 20 \
  --gp-convergence-interval 5 \
  --gp-base-seed 20280820 \
  --bdqn-max-env-steps 40000 \
  --bdqn-checkpoint-interval 5000 \
  --bdqn-min-convergence-steps 20000 \
  --bdqn-round1-seeds 10 11 12 \
  --bdqn-round2-seeds 13 14 15 \
  --bdqn-parallel-jobs 3 \
  --gp-device cpu \
  --bdqn-device cuda \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/continuation.log"

if [[ "$(sha256sum "$BASE_GP_POLICY" | awk '{print $1}')" != "$BASE_GP_SHA" ]]; then
  echo "frozen G2 policy changed during continuation" >&2
  exit 10
fi
if [[ "$(sha256sum "$BASE_BDQN_CHECKPOINT" | awk '{print $1}')" != "$BASE_BDQN_SHA" ]]; then
  echo "frozen B2 checkpoint changed during continuation" >&2
  exit 11
fi

CURRENT_STAGE=complete
update_status complete complete
trap - EXIT
