param(
    [string]$PythonPath = "D:\software\anaconda3\envs\RL_env\python.exe",
    [string]$Workspace = "E:\LocalProject\SOSRL",
    [string]$LegacyStudyManifest = "E:\LocalProject\SOSRL\runs\round1_formal\study_manifest.json",
    [string]$V2OutputDirectory = "E:\LocalProject\SOSRL\runs\round1_formal_v2",
    [int]$Workers = 8
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Workspace

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Output "[$(Get-Date -Format o)] START $Label"
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
    Write-Output "[$(Get-Date -Format o)] COMPLETE $Label"
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python runtime not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $LegacyStudyManifest)) {
    throw "Legacy study manifest not found: $LegacyStudyManifest"
}

$v2Manifest = Join-Path $V2OutputDirectory "study_manifest.json"

Invoke-PythonStep -Label "schema-v2 seed1-5 augmented study initialization" -Arguments @(
    "-u", "-m", "sosrl", "init-round1-study",
    "--augment-from", $LegacyStudyManifest,
    "--augment-seeds", "1", "2", "3", "4", "5",
    "--output-dir", $V2OutputDirectory,
    "--ss-low-threshold", "0.40",
    "--ss-high-threshold", "0.90",
    "--device", "cuda"
)

if (-not (Test-Path -LiteralPath $v2Manifest)) {
    throw "Schema-v2 study manifest was not created: $v2Manifest"
}

Invoke-PythonStep -Label "schema-v2 seed1-5 convergence supplement" -Arguments @(
    "-u", "-m", "sosrl", "run-round1-study",
    "--study-manifest", $v2Manifest,
    "--stage", "convergence",
    "--workers", $Workers.ToString()
)

Write-Output "[$(Get-Date -Format o)] ROUND1_V2_SEED1_5_CONVERGENCE_COMPLETE"
