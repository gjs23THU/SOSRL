param(
    [string]$Python = "D:\software\anaconda3\envs\RL_env\python.exe",
    [string]$SchedulerCheckpoint = "runs\branching_seed4_120000\branching_scheduler.pt",
    [string]$ManualArchitectureCheckpoint = "runs\hrl_budget20_seed4\architecture_1000\architecture.pt",
    [string]$ScenarioDir = "runs\gp_scenarios_20260820_standard",
    [string]$OutputDir = "runs\gp_architecture_standard39_seed4_bdqn120k",
    [int]$Workers = 8,
    [int]$BaseSeed = 20260820,
    [int]$PopulationSize = 120,
    [int]$Generations = 50,
    [int]$IndependentRuns = 3
)

$ErrorActionPreference = "Stop"
$PolicyPath = Join-Path $OutputDir "gp_policy.json"
$ResumePath = Join-Path $OutputDir "run_00\evolution_state.pkl"

if (-not (Test-Path -LiteralPath $PolicyPath)) {
    $TrainingArguments = @(
        "-m", "sosrl", "train-gp-architecture",
        "--scheduler-checkpoint", $SchedulerCheckpoint,
        "--scenario-dir", $ScenarioDir,
        "--feature-set", "system_delta",
        "--population-size", $PopulationSize,
        "--generations", $Generations,
        "--runs", $IndependentRuns,
        "--train-batch-size", "16",
        "--workers", $Workers,
        "--base-seed", $BaseSeed,
        "--device", "cpu",
        "--output-dir", $OutputDir
    )
    if (Test-Path -LiteralPath $ResumePath) {
        $TrainingArguments += @("--resume-state", $ResumePath)
    }
    & $Python @TrainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Standard GP training failed with exit code $LASTEXITCODE."
    }
}

foreach ($Split in @("test_iid", "test_ood")) {
    if ($Split -eq "test_iid") {
        $EvaluationName = "evaluation_iid"
    } else {
        $EvaluationName = "evaluation_ood"
    }
    $EvaluationDir = Join-Path $OutputDir $EvaluationName
    & $Python -m sosrl evaluate-gp-stack `
        --gp-policy $PolicyPath `
        --scheduler-checkpoint $SchedulerCheckpoint `
        --scenario-manifest (Join-Path $ScenarioDir "$Split.json") `
        --manual-architecture-checkpoint $ManualArchitectureCheckpoint `
        --baselines fixed random_concrete manual6_dqn gp `
        --collect-schedule `
        --device cpu `
        --output-dir $EvaluationDir
    if ($LASTEXITCODE -ne 0) {
        throw "GP stack evaluation for $Split failed with exit code $LASTEXITCODE."
    }
}

Write-Output "STANDARD_GP_EXPERIMENT_COMPLETE"
Write-Output $PolicyPath
Write-Output (Join-Path $OutputDir "evaluation_iid\evaluation_summary.csv")
Write-Output (Join-Path $OutputDir "evaluation_ood\evaluation_summary.csv")
