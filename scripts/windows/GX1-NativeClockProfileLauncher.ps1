param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanFileSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][Alias('WindowsSourceRepo')][string]$WindowsControllerSourceRoot,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExpectedControllerSha256,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$Python = '.venv/bin/python',
    [string]$CampaignControlRepo = $SourceRepo
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$controller = Join-Path $WindowsControllerSourceRoot 'scripts\windows\GX1-RandomAccessCampaignV2Controller.ps1'
if ((Get-FileHash -LiteralPath $controller -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ExpectedControllerSha256) {
    throw 'Bound campaign controller differs'
}
$controllerArguments = @{} + $PSBoundParameters
$gpu = 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
$clockLog = Join-Path $env:ProgramData 'GX1\GpuClockProfile'
[void](New-Item -ItemType Directory -Path $clockLog -Force)
$launcherSha = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$clockJob = Start-Job -ArgumentList $gpu,$clockLog,$PlanFileSha256,$ExpectedControllerSha256,$launcherSha -ScriptBlock {
    param($gpu,$clockLog,$planSha,$controllerSha,$launcherSha)
    $ErrorActionPreference = 'Stop'
    $applied = $false
    try {
        while ($true) {
            $raw = @(& nvidia-smi.exe -i $gpu --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
            if ($LASTEXITCODE -ne 0 -or $raw.Count -ne 1) { throw 'GPU workload sample unavailable' }
            $values = @($raw[0].Split(',') | ForEach-Object { [int]$_.Trim() })
            # Match the unchanged keeper's 384 MiB idle boundary. Do not lock
            # clocks during CPU startup, physical reboot preparation or idle.
            if (-not $applied -and $values[0] -gt 384 -and $values[1] -gt 0) {
                & nvidia-smi.exe -i $gpu -lgc 1395,1695 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw 'GPU clock setup failed' }
                & nvidia-smi.exe -i $gpu -lmc 9751 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw 'GPU memory clock setup failed' }
                $applied = $true
                $record = [ordered]@{
                    observed_utc = [DateTimeOffset]::UtcNow.ToString('o')
                    boot_id = [long](Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters' -Name BootId).BootId
                    plan_file_sha256 = $planSha
                    gpu_uuid = $gpu
                    graphics_mhz = @(1395,1695)
                    requested_memory_mhz = 9751
                    controller_sha256 = $controllerSha
                    launcher_sha256 = $launcherSha
                    workload_memory_mib = $values[0]
                    workload_utilization_percent = $values[1]
                }
                [IO.File]::AppendAllText((Join-Path $clockLog 'APPLIED.jsonl'), (($record | ConvertTo-Json -Depth 3 -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
            }
            elseif ($applied -and $values[0] -le 384) {
                & nvidia-smi.exe -i $gpu -rgc | Out-Null
                & nvidia-smi.exe -i $gpu -rmc | Out-Null
                $applied = $false
            }
            Start-Sleep -Seconds 5
        }
    }
    finally {
        & nvidia-smi.exe -i $gpu -rgc | Out-Null
        & nvidia-smi.exe -i $gpu -rmc | Out-Null
    }
}
$controllerExit = 0
try {
    & $controller @controllerArguments
    $controllerExit = $LASTEXITCODE
}
finally {
    Stop-Job -Job $clockJob -ErrorAction SilentlyContinue
    Remove-Job -Job $clockJob -Force -ErrorAction SilentlyContinue
    & nvidia-smi.exe -i $gpu -rgc | Out-Null
    & nvidia-smi.exe -i $gpu -rmc | Out-Null
}
exit $controllerExit
