#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PYTHON ROUND1_SCENARIO_ROOT G0_POLICY B0_CHECKPOINT OUTPUT_ROOT EXPECTED_COMMIT" >&2
  exit 2
fi

PYTHON_BIN=$1
ROUND1_ROOT=$2
G0_POLICY=$3
B0_CHECKPOINT=$4
OUTPUT_ROOT=$5
EXPECTED_COMMIT=$6
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCENARIO_ROOT="$OUTPUT_ROOT/scenarios"
SMOKE_SCENARIOS="$SCENARIO_ROOT/smoke"
FORMAL_SCENARIOS="$SCENARIO_ROOT/formal"
SMOKE_OUTPUT="$OUTPUT_ROOT/smoke_8x10x1_5k"
FORMAL_OUTPUT="$OUTPUT_ROOT/formal_120x50x3_40k"
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
  echo "another alternation pipeline holds $OUTPUT_ROOT/pipeline.lock" >&2
  exit 3
fi

EXISTING_MANIFESTS=(
  "$ROUND1_ROOT/b/train.json"
  "$ROUND1_ROOT/b/validation.json"
  "$ROUND1_ROOT/g/train.json"
  "$ROUND1_ROOT/g/validation.json"
  "$TEST_IID_MANIFEST"
  "$TEST_OOD_MANIFEST"
)
for required in "$PYTHON_BIN" "$G0_POLICY" "$B0_CHECKPOINT" "${EXISTING_MANIFESTS[@]}"; do
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
G0_SHA=$(sha256sum "$G0_POLICY" | awk '{print $1}')
B0_SHA=$(sha256sum "$B0_CHECKPOINT" | awk '{print $1}')
if [[ -n "${SOSRL_EXPECTED_G0_SHA:-}" && "$G0_SHA" != "$SOSRL_EXPECTED_G0_SHA" ]]; then
  echo "G0 SHA mismatch: expected $SOSRL_EXPECTED_G0_SHA, got $G0_SHA" >&2
  exit 8
fi
if [[ -n "${SOSRL_EXPECTED_B0_SHA:-}" && "$B0_SHA" != "$SOSRL_EXPECTED_B0_SHA" ]]; then
  echo "B0 SHA mismatch: expected $SOSRL_EXPECTED_B0_SHA, got $B0_SHA" >&2
  exit 9
fi
"$PYTHON_BIN" -c 'import torch,deap; assert torch.cuda.is_available(), "CUDA is required for BDQN"; print({"torch":torch.__version__,"cuda_device":torch.cuda.get_device_name(0),"deap":deap.__version__})'
"$PYTHON_BIN" -c 'import sys; from sosrl.workflows.gp_architecture import load_scenario_manifest; [(lambda m,p: print(p,m["manifest_hash"],m["size"]))(load_scenario_manifest(p),p) for p in sys.argv[1:]]' "${EXISTING_MANIFESTS[@]}"
sha256sum "$G0_POLICY" "$B0_CHECKPOINT" "${EXISTING_MANIFESTS[@]}" | tee "$OUTPUT_ROOT/logs/input_sha256.txt"
update_status preflight complete "g0_sha=$G0_SHA b0_sha=$B0_SHA"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CURRENT_STAGE=tests
update_status tests running
"$PYTHON_BIN" -m pytest -q \
  tests/test_alternating_stack.py \
  tests/test_gp_evolution.py \
  tests/test_gp_workflow.py \
  tests/test_round1_study.py \
  tests/test_cli.py \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/tests.log"
update_status tests complete

SCENARIO_ARGS=()
for manifest in "${EXISTING_MANIFESTS[@]}"; do
  SCENARIO_ARGS+=(--existing-manifest "$manifest")
done

CURRENT_STAGE=scenario_generation
update_status scenario_generation running
if [[ ! -f "$SMOKE_SCENARIOS/scenario_registry.json" ]]; then
  "$PYTHON_BIN" -m sosrl generate-alternation-scenarios \
    "${SCENARIO_ARGS[@]}" \
    --gate-iid-size 4 --gate-ood-size 4 \
    --final-iid-size 4 --final-ood-size 4 \
    --output-dir "$SMOKE_SCENARIOS"
fi
if [[ ! -f "$FORMAL_SCENARIOS/scenario_registry.json" ]]; then
  "$PYTHON_BIN" -m sosrl generate-alternation-scenarios \
    "${SCENARIO_ARGS[@]}" \
    --output-dir "$FORMAL_SCENARIOS"
fi
sha256sum "$SMOKE_SCENARIOS"/*.json "$FORMAL_SCENARIOS"/*.json \
  | tee "$OUTPUT_ROOT/logs/alternation_scenario_sha256.txt"
update_status scenario_generation complete

CURRENT_STAGE=smoke
update_status smoke running
SOSRL_ALLOW_RESUME=1 "$PYTHON_BIN" -u -m sosrl run-gp-bdqn-alternation \
  --base-gp-policy "$G0_POLICY" \
  --base-scheduler-checkpoint "$B0_CHECKPOINT" \
  --scenario-dir "$ROUND1_ROOT/g" \
  --gate-iid-manifest "$SMOKE_SCENARIOS/gate_iid.json" \
  --gate-ood-manifest "$SMOKE_SCENARIOS/gate_ood.json" \
  --final-iid-manifest "$SMOKE_SCENARIOS/final_iid.json" \
  --final-ood-manifest "$SMOKE_SCENARIOS/final_ood.json" \
  --output-dir "$SMOKE_OUTPUT" \
  --gp-population-size 8 \
  --gp-max-generations 10 \
  --gp-runs 1 \
  --gp-workers 1 \
  --gp-min-generations 10 \
  --gp-convergence-interval 5 \
  --bdqn-max-env-steps 5000 \
  --bdqn-checkpoint-interval 5000 \
  --bdqn-min-convergence-steps 5000 \
  --bdqn-parallel-jobs 3 \
  --gp-device cpu \
  --bdqn-device cuda \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/smoke.log"
update_status smoke complete

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
  echo "formal alternation waiting for resources: $RESOURCE_DETAILS"
  sleep 300
done

CURRENT_STAGE=formal
update_status formal running
SOSRL_ALLOW_RESUME=1 "$PYTHON_BIN" -u -m sosrl run-gp-bdqn-alternation \
  --base-gp-policy "$G0_POLICY" \
  --base-scheduler-checkpoint "$B0_CHECKPOINT" \
  --scenario-dir "$ROUND1_ROOT/g" \
  --gate-iid-manifest "$FORMAL_SCENARIOS/gate_iid.json" \
  --gate-ood-manifest "$FORMAL_SCENARIOS/gate_ood.json" \
  --final-iid-manifest "$FORMAL_SCENARIOS/final_iid.json" \
  --final-ood-manifest "$FORMAL_SCENARIOS/final_ood.json" \
  --output-dir "$FORMAL_OUTPUT" \
  --gp-population-size 120 \
  --gp-max-generations 50 \
  --gp-runs 3 \
  --gp-workers 12 \
  --gp-min-generations 20 \
  --gp-convergence-interval 5 \
  --bdqn-max-env-steps 40000 \
  --bdqn-checkpoint-interval 5000 \
  --bdqn-min-convergence-steps 20000 \
  --bdqn-parallel-jobs 3 \
  --gp-device cpu \
  --bdqn-device cuda \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/formal.log"

CURRENT_STAGE=complete
update_status complete complete
trap - EXIT
