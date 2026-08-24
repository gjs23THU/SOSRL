param(
    [ValidateRange(1, 8)]
    [int]$Seed = 8,
    [ValidateSet("fixed", "arch", "g0")]
    [string[]]$Providers = @("fixed", "arch", "g0"),
    [string]$PythonPath = "D:\software\anaconda3\envs\RL_env\python.exe",
    [string]$Workspace = "E:\LocalProject\SOSRL",
    [string]$StudyRoot = "E:\LocalProject\SOSRL\runs\round1_formal",
    [int]$MinimumGpuFreeMiB = 3800,
    [int]$MaximumCpuPercent = 70,
    [int]$MinimumAvailableMemoryMiB = 4096,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Workspace

function Get-ResourceSnapshot {
    $samples = Get-Counter `
        "\Processor(_Total)\% Processor Time", `
        "\Memory\Available MBytes" `
        -SampleInterval 1 `
        -MaxSamples 1
    $cpu = ($samples.CounterSamples | Where-Object Path -Like "*processor(_total)*").CookedValue
    $availableMemory = (
        $samples.CounterSamples | Where-Object Path -Like "*memory\available mbytes"
    ).CookedValue
    $gpuFreeText = & nvidia-smi `
        --query-gpu=memory.free `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed while reading free GPU memory."
    }
    $gpuFree = [int](($gpuFreeText | Select-Object -First 1).Trim())
    return [pscustomobject]@{
        CpuPercent = [math]::Round([double]$cpu, 2)
        AvailableMemoryMiB = [math]::Round([double]$availableMemory, 0)
        GpuFreeMiB = $gpuFree
    }
}

function Wait-ForResources {
    while ($true) {
        $snapshot = Get-ResourceSnapshot
        $timestamp = Get-Date -Format o
        Write-Output (
            "[$timestamp] RESOURCE cpu={0}% ram_free={1}MiB gpu_free={2}MiB" -f `
                $snapshot.CpuPercent,
                $snapshot.AvailableMemoryMiB,
                $snapshot.GpuFreeMiB
        )
        if (
            $snapshot.CpuPercent -le $MaximumCpuPercent -and
            $snapshot.AvailableMemoryMiB -ge $MinimumAvailableMemoryMiB -and
            $snapshot.GpuFreeMiB -ge $MinimumGpuFreeMiB
        ) {
            return
        }
        Write-Output "[$timestamp] WAIT resource guard not satisfied"
        Start-Sleep -Seconds 30
    }
}

$studyManifestPath = Join-Path $StudyRoot "study_manifest.json"
if (-not (Test-Path -LiteralPath $studyManifestPath)) {
    throw "Study manifest not found: $studyManifestPath"
}
$study = Get-Content -LiteralPath $studyManifestPath -Raw | ConvertFrom-Json
if ([int]$study.schema_version -ne 1) {
    throw "Parallel legacy seed runner requires a schema-v1 study."
}
if (-not [bool]$study.test_locked) {
    throw "Test-v2 must remain locked during convergence."
}

$sourceCheckpoint = Join-Path $StudyRoot "initial_weights\seed_$Seed.pt"
if (-not (Test-Path -LiteralPath $sourceCheckpoint)) {
    throw "Initial checkpoint not found: $sourceCheckpoint"
}

$checkpointSteps = @(
    "0", "20000", "40000", "60000", "80000", "120000", "160000", "200000"
)
foreach ($provider in $Providers) {
    $cellDirectory = Join-Path $StudyRoot "bdqn\convergence\$provider\seed_$Seed"
    $cellManifest = Join-Path $cellDirectory "cell_manifest.json"
    if (Test-Path -LiteralPath $cellManifest) {
        $cell = Get-Content -LiteralPath $cellManifest -Raw | ConvertFrom-Json
        if ($cell.status -eq "complete") {
            Write-Output "[$(Get-Date -Format o)] SKIP complete $provider seed=$Seed"
            continue
        }
    }

    Wait-ForResources
    $arguments = @(
        "-u", "-m", "sosrl", "train-round1-bdqn-cell",
        "--provider", $provider,
        "--mode", "scratch",
        "--source-checkpoint", $sourceCheckpoint,
        "--architecture-checkpoint", [string]$study.inputs.architecture_checkpoint.path,
        "--gp-policy", [string]$study.inputs.g0_policy.path,
        "--train-manifest", [string]$study.scenarios.b_train,
        "--validation-manifest", [string]$study.scenarios.b_validation,
        "--output-dir", $cellDirectory,
        "--seed", $Seed.ToString(),
        "--max-env-steps", "200000",
        "--device", "cuda",
        "--checkpoint-steps"
    ) + $checkpointSteps

    Write-Output "[$(Get-Date -Format o)] START $provider seed=$Seed"
    if ($DryRun) {
        Write-Output "$PythonPath $($arguments -join ' ')"
        continue
    }
    & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$provider seed=$Seed failed with exit code $LASTEXITCODE."
    }
    Write-Output "[$(Get-Date -Format o)] COMPLETE $provider seed=$Seed"
}

Write-Output "[$(Get-Date -Format o)] PARALLEL_SEED_BRANCH_COMPLETE seed=$Seed"
