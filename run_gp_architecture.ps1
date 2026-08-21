param(
    [string]$Python = "D:\software\anaconda3\envs\RL_env\python.exe",
    [string]$SchedulerCheckpoint = "runs\branching_scheduler\branching_scheduler.pt",
    [string]$ManualArchitectureCheckpoint = "",
    [string]$ScenarioDir = "runs\gp_scenarios",
    [string]$OutputDir = "runs\gp_architecture",
    [int]$Workers = 8,
    [int]$BaseSeed = 20260820
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $SchedulerCheckpoint)) {
    throw "Frozen Branching DQN checkpoint not found: $SchedulerCheckpoint"
}

& $Python -m sosrl generate-gp-scenarios `
    --base-seed $BaseSeed `
    --train-size 256 `
    --validation-size 128 `
    --test-size 500 `
    --ood-size 200 `
    --output-dir $ScenarioDir

& $Python -m sosrl train-gp-architecture `
    --scheduler-checkpoint $SchedulerCheckpoint `
    --scenario-dir $ScenarioDir `
    --feature-set system_delta `
    --population-size 200 `
    --generations 80 `
    --runs 10 `
    --train-batch-size 16 `
    --anchor-size 64 `
    --base-seed $BaseSeed `
    --workers $Workers `
    --device cpu `
    --output-dir $OutputDir

$EvaluationBaselines = @("fixed", "random_concrete", "gp")
$ManualCheckpointArguments = @()
if ($ManualArchitectureCheckpoint) {
    if (-not (Test-Path -LiteralPath $ManualArchitectureCheckpoint)) {
        throw "Manual architecture checkpoint not found: $ManualArchitectureCheckpoint"
    }
    $EvaluationBaselines = @("fixed", "random_concrete", "manual6_dqn", "gp")
    $ManualCheckpointArguments = @(
        "--manual-architecture-checkpoint",
        $ManualArchitectureCheckpoint
    )
}

& $Python -m sosrl evaluate-gp-stack `
    --gp-policy "$OutputDir\gp_policy.json" `
    --scheduler-checkpoint $SchedulerCheckpoint `
    --scenario-manifest "$ScenarioDir\test_iid.json" `
    --baselines $EvaluationBaselines `
    @ManualCheckpointArguments `
    --collect-schedule `
    --device cpu `
    --output-dir "$OutputDir\evaluation_iid"

& $Python -m sosrl evaluate-gp-stack `
    --gp-policy "$OutputDir\gp_policy.json" `
    --scheduler-checkpoint $SchedulerCheckpoint `
    --scenario-manifest "$ScenarioDir\test_ood.json" `
    --baselines $EvaluationBaselines `
    @ManualCheckpointArguments `
    --collect-schedule `
    --device cpu `
    --output-dir "$OutputDir\evaluation_ood"
