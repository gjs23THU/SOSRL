param(
    [int[]]$Seeds = @(1, 2, 3, 4, 5, 6),

    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$python = "D:\software\anaconda3\envs\RL_env\python.exe"
$seedRunner = Join-Path $repoRoot "run_flat_rules_seed.ps1"
$outputDir = Join-Path $repoRoot "runs\flat_rules_multiseed_report"

Set-Location -LiteralPath $repoRoot
foreach ($seed in $Seeds) {
    & $seedRunner -Seed $seed -Device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Flat-rule seed $seed failed with code $LASTEXITCODE."
    }
}

$reportArgs = @(
    "-u", "-m", "scripts.report_flat_rule_comparison",
    "--runs-root", (Join-Path $repoRoot "runs"),
    "--output-dir", $outputDir
)
foreach ($seed in $Seeds) {
    $reportArgs += @("--seed", $seed.ToString())
}
& $python @reportArgs
if ($LASTEXITCODE -ne 0) {
    throw "Flat-rule multiseed report failed with code $LASTEXITCODE."
}
