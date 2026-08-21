param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$Seed,

    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$python = "D:\software\anaconda3\envs\RL_env\python.exe"
$schedulerHistory = Join-Path $repoRoot "runs\SIG1000_standard_seed4\train_history.csv"
$hrlRoot = Join-Path $repoRoot ("runs\hrl_budget20_seed{0}" -f $Seed)
$hrlHistory = Join-Path $hrlRoot "architecture_1000\architecture_history.csv"
$hrlCheckpoint = Join-Path $hrlRoot "architecture_1000\hrl.pt"
$runRoot = Join-Path $repoRoot ("runs\flat_rules_budgetmatched_seed{0}" -f $Seed)
$trainDir = Join-Path $runRoot "flat_rules_128"
$evalDir = Join-Path $runRoot "evaluation_flat_rules_128"
$statusPath = Join-Path $runRoot "status.json"
$trainLog = Join-Path $runRoot "train.log"
$evalLog = Join-Path $runRoot "evaluate.log"

function Get-AssignedOperationCount {
    param([Parameter(Mandatory = $true)][string]$HistoryPath)

    if (-not (Test-Path -LiteralPath $HistoryPath)) {
        throw "Training history does not exist: $HistoryPath"
    }
    $measurement = Import-Csv -LiteralPath $HistoryPath |
        Measure-Object -Property assigned_ops -Sum
    return [int64]$measurement.Sum
}

function Write-RunStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [string]$Message = "",
        [int64]$TargetEnvironmentSteps = 0
    )

    [ordered]@{
        seed = $Seed
        algorithm = "flat_rule_dqn"
        hidden_dim = 128
        stage = $Stage
        message = $Message
        target_environment_steps = $TargetEnvironmentSteps
        process_id = $PID
        updated_at = (Get-Date).ToString("o")
        hrl_checkpoint = $hrlCheckpoint
        train_dir = $trainDir
        evaluation_dir = $evalDir
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "RL Python interpreter does not exist: $python"
}
if (-not (Test-Path -LiteralPath $hrlCheckpoint)) {
    throw "HRL checkpoint does not exist: $hrlCheckpoint"
}

$schedulerSteps = Get-AssignedOperationCount -HistoryPath $schedulerHistory
$architectureSteps = Get-AssignedOperationCount -HistoryPath $hrlHistory
$targetEnvironmentSteps = $schedulerSteps + $architectureSteps

New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
Set-Location -LiteralPath $repoRoot

try {
    Write-RunStatus -Stage "training" -TargetEnvironmentSteps $targetEnvironmentSteps
    $trainArgs = @(
        "-u", "-m", "sosrl", "train-flat-rules",
        "--scenario-pool-size", "100",
        "--max-env-steps", $targetEnvironmentSteps.ToString(),
        "--budget", "8000",
        "--refund-rate", "0.8",
        "--gamma", "0.99",
        "--n-step", "5",
        "--lr", "0.0001",
        "--batch-size", "64",
        "--buffer-size", "50000",
        "--min-buffer-size", "500",
        "--target-update-interval", "100",
        "--epsilon-start", "1.0",
        "--epsilon-end", "0.05",
        "--epsilon-decay", "0.995",
        "--hidden-dim", "128",
        "--seed", $Seed.ToString(),
        "--device", $Device,
        "--output-dir", $trainDir
    )
    & $python @trainArgs *> $trainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Flat-rule training exited with code $LASTEXITCODE."
    }

    Write-RunStatus -Stage "evaluating" -TargetEnvironmentSteps $targetEnvironmentSteps
    $flatCheckpoint = Join-Path $trainDir "flat_rules.pt"
    $evalArgs = @(
        "-u", "-m", "sosrl", "evaluate",
        "--checkpoint", $hrlCheckpoint,
        "--flat-rule-model", ("flat_rule_dqn={0}" -f $flatCheckpoint),
        "--eval-episodes", "100",
        "--eval-seed", "20260724",
        "--device", $Device,
        "--output-dir", $evalDir
    )
    & $python @evalArgs *> $evalLog
    if ($LASTEXITCODE -ne 0) {
        throw "Paired evaluation exited with code $LASTEXITCODE."
    }

    Write-RunStatus -Stage "complete" -TargetEnvironmentSteps $targetEnvironmentSteps
}
catch {
    Write-RunStatus `
        -Stage "failed" `
        -Message $_.Exception.Message `
        -TargetEnvironmentSteps $targetEnvironmentSteps
    $_ | Out-String | Add-Content -LiteralPath $trainLog -Encoding utf8
    exit 1
}
