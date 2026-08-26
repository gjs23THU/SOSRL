param(
    [switch]$ProbeOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspace = 'E:\LocalProject\SOSRL'
$python = 'D:\software\anaconda3\envs\RL_env\python.exe'
$outputDir = 'E:\LocalProject\SOSRL\runs\round1_fixed_tuning\lr_decay_0p9975_to320k\seed_1'
$logDir = 'E:\LocalProject\SOSRL\runs\round1_fixed_tuning\logs'
$trainStdout = Join-Path $logDir 'fixed_lr_decay_0p9975_to320k_seed1.stdout.log'
$trainStderr = Join-Path $logDir 'fixed_lr_decay_0p9975_to320k_seed1.stderr.log'
$studyManifest = 'E:\LocalProject\SOSRL\runs\round1_formal_v2\study_manifest.json'

$minimumAvailableRamMb = 6144.0
$minimumAvailableGpuMb = 3584.0
$maximumCpuPercent = 65.0
$pollSeconds = 60

function Get-ResourceSnapshot {
    $availableRamMb = [double](
        Get-Counter '\Memory\Available MBytes'
    ).CounterSamples.CookedValue
    $cpuPercent = [double](
        Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1
    ).CounterSamples.CookedValue
    $gpuLine = (& nvidia-smi `
        --query-gpu=memory.used,memory.total `
        --format=csv,noheader,nounits | Select-Object -First 1)
    if (-not $gpuLine) {
        throw 'nvidia-smi did not return GPU memory information.'
    }
    $gpuParts = $gpuLine.Split(',')
    $gpuUsedMb = [double]$gpuParts[0].Trim()
    $gpuTotalMb = [double]$gpuParts[1].Trim()
    [pscustomobject]@{
        AvailableRamMb = $availableRamMb
        CpuPercent = $cpuPercent
        GpuUsedMb = $gpuUsedMb
        GpuTotalMb = $gpuTotalMb
        GpuFreeMb = $gpuTotalMb - $gpuUsedMb
    }
}

function Write-Snapshot {
    param($Snapshot)
    $timestamp = [DateTimeOffset]::Now.ToString('o')
    Write-Output (
        'RESOURCE {0} ram_available_mb={1:N0} cpu_percent={2:N1} gpu_used_mb={3:N0} gpu_free_mb={4:N0}' -f `
        $timestamp,
        $Snapshot.AvailableRamMb,
        $Snapshot.CpuPercent,
        $Snapshot.GpuUsedMb,
        $Snapshot.GpuFreeMb
    )
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "RL Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $studyManifest)) {
    throw "Study manifest is missing: $studyManifest"
}
$study = Get-Content -LiteralPath $studyManifest -Raw | ConvertFrom-Json
if ($study.test_locked -ne $true) {
    throw 'Test-v2 is not locked; refusing to launch the tuning cell.'
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

while ($true) {
    if (Test-Path -LiteralPath $outputDir) {
        Write-Output "OUTPUT_EXISTS refusing duplicate launch: $outputDir"
        exit 2
    }

    $snapshot = Get-ResourceSnapshot
    Write-Snapshot $snapshot
    $ready = (
        $snapshot.AvailableRamMb -ge $minimumAvailableRamMb -and
        $snapshot.GpuFreeMb -ge $minimumAvailableGpuMb -and
        $snapshot.CpuPercent -le $maximumCpuPercent
    )

    if ($ProbeOnly) {
        Write-Output (
            'PROBE_ONLY ready={0} required_ram_mb={1:N0} required_gpu_free_mb={2:N0} maximum_cpu_percent={3:N1}' -f `
            $ready,
            $minimumAvailableRamMb,
            $minimumAvailableGpuMb,
            $maximumCpuPercent
        )
        exit 0
    }

    if ($ready) {
        $arguments = @(
            '-m', 'sosrl', 'train-round1-bdqn-cell',
            '--provider', 'fixed',
            '--mode', 'scratch',
            '--source-checkpoint', 'E:\LocalProject\SOSRL\runs\round1_formal\initial_weights\seed_1.pt',
            '--train-manifest', 'E:\LocalProject\SOSRL\runs\round1_formal\scenarios\b\train.json',
            '--validation-manifest', 'E:\LocalProject\SOSRL\runs\round1_formal\scenarios\b\validation.json',
            '--output-dir', $outputDir,
            '--seed', '1',
            '--max-env-steps', '320000',
            '--checkpoint-steps', '0', '20000', '40000', '60000', '80000', '120000', '160000', '200000', '240000', '280000', '320000',
            '--lr', '0.0001',
            '--lr-end', '0.00001',
            '--lr-decay', '0.9975',
            '--batch-size', '64',
            '--buffer-size', '50000',
            '--target-update-interval', '250',
            '--epsilon-start', '1.0',
            '--epsilon-end', '0.05',
            '--epsilon-decay', '0.995',
            '--device', 'cuda'
        )
        $process = Start-Process `
            -FilePath $python `
            -ArgumentList $arguments `
            -WorkingDirectory $workspace `
            -RedirectStandardOutput $trainStdout `
            -RedirectStandardError $trainStderr `
            -WindowStyle Hidden `
            -PassThru
        Write-Output ('TRAIN_PID=' + $process.Id)
        Write-Output ('OUTPUT=' + $outputDir)
        Write-Output ('TRAIN_STDOUT=' + $trainStdout)
        Write-Output ('TRAIN_STDERR=' + $trainStderr)
        exit 0
    }

    Start-Sleep -Seconds $pollSeconds
}
