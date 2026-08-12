param(
    [Parameter(Mandatory = $true)]
    [int]$Seed
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$python = "D:\software\anaconda3\envs\RL_env\python.exe"
$schedulerCheckpoint = Join-Path $repoRoot "runs\SIG1000_standard_seed4\model.pt"
$runRoot = Join-Path $repoRoot ("runs\hrl_budget20_seed{0}" -f $Seed)
$trainDir = Join-Path $runRoot "architecture_1000"
$evalDir = Join-Path $runRoot "evaluation_architecture_1000"
$statusPath = Join-Path $runRoot "status.json"
$trainLog = Join-Path $runRoot "train.log"
$evalLog = Join-Path $runRoot "evaluate.log"

function Write-RunStatus {
    param(
        [string]$Stage,
        [string]$Message = ""
    )

    [ordered]@{
        seed = $Seed
        stage = $Stage
        message = $Message
        process_id = $PID
        updated_at = (Get-Date).ToString("o")
        train_dir = $trainDir
        evaluation_dir = $evalDir
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
Set-Location -LiteralPath $repoRoot

try {
    Write-RunStatus -Stage "training"
    $trainArgs = @(
        "-u", ".\hrlmain.py", "train-architecture",
        "--scheduler-checkpoint", $schedulerCheckpoint,
        "--episodes", "1000",
        "--scenario-pool-size", "100",
        "--budget", "8000",
        "--refund-rate", "0.8",
        "--gamma", "0.99",
        "--n-step", "5",
        "--architecture-lr", "0.0001",
        "--batch-size", "64",
        "--buffer-size", "50000",
        "--min-buffer-size", "500",
        "--target-update-interval", "100",
        "--epsilon-start", "1.0",
        "--epsilon-end", "0.05",
        "--epsilon-decay", "0.995",
        "--hidden-dim", "128",
        "--seed", $Seed.ToString(),
        "--device", "cuda",
        "--output-dir", $trainDir
    )
    & $python @trainArgs *> $trainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Architecture training exited with code $LASTEXITCODE."
    }

    Write-RunStatus -Stage "evaluating"
    $evalArgs = @(
        "-u", ".\hrlmain.py", "evaluate",
        "--checkpoint", (Join-Path $trainDir "hrl.pt"),
        "--eval-episodes", "100",
        "--eval-seed", "20260724",
        "--device", "cuda",
        "--output-dir", $evalDir
    )
    & $python @evalArgs *> $evalLog
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation exited with code $LASTEXITCODE."
    }

    Write-RunStatus -Stage "complete"
}
catch {
    Write-RunStatus -Stage "failed" -Message $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath $trainLog -Encoding utf8
    exit 1
}
