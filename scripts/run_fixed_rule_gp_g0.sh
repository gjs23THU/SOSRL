#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PYTHON ROUND1_SCENARIO_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

PYTHON_BIN=$1
SCENARIO_ROOT=$2
OUTPUT_ROOT=$3
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_MANIFEST="$SCENARIO_ROOT/b/train.json"
VALIDATION_MANIFEST="$SCENARIO_ROOT/b/validation.json"
GP_SCENARIO_DIR="$SCENARIO_ROOT/g"
SCHEDULER_OUTPUT="$OUTPUT_ROOT/fixed_rule_scheduler"
SMOKE_OUTPUT="$OUTPUT_ROOT/gp_smoke_8x2x1"
FORMAL_OUTPUT="$OUTPUT_ROOT/gp_formal_120x50x3"
STATUS_PATH="$OUTPUT_ROOT/pipeline_status.json"

if [[ -e "$OUTPUT_ROOT" && "${SOSRL_ALLOW_RESUME:-0}" != "1" ]]; then
  echo "output root already exists; set SOSRL_ALLOW_RESUME=1 to resume: $OUTPUT_ROOT" >&2
  exit 5
fi
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "another fixed-rule GP pipeline holds $OUTPUT_ROOT/pipeline.lock" >&2
  exit 3
fi

for required in "$PYTHON_BIN" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" \
  "$GP_SCENARIO_DIR/train.json" "$GP_SCENARIO_DIR/validation.json"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 4
  fi
done

update_status() {
  local stage=$1
  local state=$2
  "$PYTHON_BIN" -c 'import json, pathlib, sys, datetime; p=pathlib.Path(sys.argv[1]); payload={"stage":sys.argv[2],"status":sys.argv[3],"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_commit":sys.argv[4],"pid":int(sys.argv[5])}; p.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")' \
    "$STATUS_PATH" "$stage" "$state" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$$"
}

CURRENT_STAGE=preflight
record_failure() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    update_status "$CURRENT_STAGE" failed || true
  fi
  return "$exit_code"
}
trap record_failure EXIT

update_status preflight running
ACTUAL_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
EXPECTED_COMMIT=${SOSRL_EXPECTED_COMMIT:-$ACTUAL_COMMIT}
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "code SHA mismatch: expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 6
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "training worktree is not clean: $REPO_ROOT" >&2
  exit 7
fi
AVAILABLE_KB=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
MIN_FREE_KB=${SOSRL_MIN_FREE_KB:-10485760}
if (( AVAILABLE_KB < MIN_FREE_KB )); then
  echo "insufficient disk space: ${AVAILABLE_KB} KiB available" >&2
  exit 8
fi
echo "code_commit=$ACTUAL_COMMIT available_disk_kib=$AVAILABLE_KB"
"$PYTHON_BIN" -c 'import torch, deap; assert torch.cuda.is_available(), "CUDA is required for rule-DQN training"; print({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0),"deap":deap.__version__})'
"$PYTHON_BIN" -c 'import sys; from sosrl.workflows.gp_architecture import load_scenario_manifest; [(lambda payload, path: print(path, payload["manifest_hash"], len(payload["scenarios"])))(load_scenario_manifest(path), path) for path in sys.argv[1:]]' \
  "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" \
  "$GP_SCENARIO_DIR/train.json" "$GP_SCENARIO_DIR/validation.json"
sha256sum "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" \
  "$GP_SCENARIO_DIR/train.json" "$GP_SCENARIO_DIR/validation.json"
update_status preflight complete

CURRENT_STAGE=fixed_rule_scheduler
update_status fixed_rule_scheduler running
"$PYTHON_BIN" -u -m sosrl train-fixed-rule-scheduler \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --output-dir "$SCHEDULER_OUTPUT" \
  --max-env-steps 200000 \
  --checkpoint-steps 0 20000 40000 60000 80000 120000 160000 200000 \
  --seed 4 \
  --device cuda \
  2>&1 | tee -a "$OUTPUT_ROOT/logs/fixed_rule_scheduler.log"
update_status fixed_rule_scheduler complete

SELECTED_CHECKPOINT=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["checkpoint"])' "$SCHEDULER_OUTPUT/validation/selection.json")
sha256sum "$SELECTED_CHECKPOINT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CURRENT_STAGE=gp_smoke
update_status gp_smoke running
if [[ ! -f "$SMOKE_OUTPUT/gp_policy.json" ]]; then
  "$PYTHON_BIN" -u -m sosrl train-gp-architecture \
    --scheduler-backend rule-dqn \
    --scheduler-checkpoint "$SELECTED_CHECKPOINT" \
    --scenario-dir "$GP_SCENARIO_DIR" \
    --feature-set system_delta \
    --population-size 8 \
    --generations 2 \
    --runs 1 \
    --train-batch-size 4 \
    --anchor-size 4 \
    --anchor-interval 1 \
    --anchor-top-k 2 \
    --workers 2 \
    --base-seed 20260820 \
    --device cpu \
    --output-dir "$SMOKE_OUTPUT" \
    2>&1 | tee -a "$OUTPUT_ROOT/logs/gp_smoke.log"
fi
update_status gp_smoke complete

CURRENT_STAGE=gp_formal
update_status gp_formal running
if [[ ! -f "$FORMAL_OUTPUT/gp_policy.json" ]]; then
  "$PYTHON_BIN" -u -m sosrl train-gp-architecture \
    --scheduler-backend rule-dqn \
    --scheduler-checkpoint "$SELECTED_CHECKPOINT" \
    --scenario-dir "$GP_SCENARIO_DIR" \
    --feature-set system_delta \
    --population-size 120 \
    --generations 50 \
    --runs 3 \
    --train-batch-size 16 \
    --anchor-size 64 \
    --anchor-interval 10 \
    --anchor-top-k 10 \
    --workers 12 \
    --base-seed 20260820 \
    --device cpu \
    --output-dir "$FORMAL_OUTPUT" \
    2>&1 | tee -a "$OUTPUT_ROOT/logs/gp_formal.log"
fi
CURRENT_STAGE=complete
update_status complete complete
