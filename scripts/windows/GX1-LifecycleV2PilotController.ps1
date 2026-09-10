param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][string]$WindowsSourceRepo,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$Python = '.venv/bin/python',
    [string]$NvidiaSmiPath = 'C:\Windows\System32\nvidia-smi.exe'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# This coordinator is observation-only for GPU power. It contains no power setter.
function Invoke-Gx1PilotJson {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $lines = @(& wsl.exe -d $Distro -u $LinuxUser --cd $SourceRepo -- $Python -m gx1.scripts.local_lifecycle_v2_pilot_campaign_v1 @Arguments)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1) { throw 'Pilot coordinator gate failed' }
    return ($lines[0] | ConvertFrom-Json)
}
function Convert-Gx1WslPath {
    param([Parameter(Mandatory = $true)][string]$LinuxPath)
    $value = @(& wsl.exe -d $Distro -u $LinuxUser -- wslpath -w $LinuxPath)
    if ($LASTEXITCODE -ne 0 -or $value.Count -ne 1) { throw 'WSL path conversion failed' }
    return $value[0]
}
function Get-Gx1Gpu {
    $raw = @(& $NvidiaSmiPath --query-gpu=uuid,power.limit,power.draw --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $raw.Count -ne 1) { throw 'GPU sample unavailable' }
    $v = @($raw[0].Split(',') | ForEach-Object { $_.Trim() })
    return [pscustomobject]@{
        uuid = $v[0]
        limit = [double]::Parse($v[1], [Globalization.CultureInfo]::InvariantCulture)
        draw = [double]::Parse($v[2], [Globalization.CultureInfo]::InvariantCulture)
    }
}
$bootUtc = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
$gpu = Get-Gx1Gpu
if ($gpu.limit -ne 160.0 -or $gpu.draw -gt 160.0) { throw 'Physical GPU is not within exact 160 W policy' }
$status = Invoke-Gx1PilotJson -Arguments @('inspect', '--plan-json', $PlanJson, '--plan-sha256', $PlanSha256, '--current-windows-boot-utc', $bootUtc)
$action = $status.action
if ($action.decision -ceq 'REBOOT_REQUIRED') {
    & shutdown.exe /r /t 30 /d p:0:0 /c 'GX1 lifecycle-v2 pilot requires a fresh boot before the next heavy invocation'
    exit 0
}
if ($action.decision -ceq 'COMPLETE' -or $action.decision -like 'BLOCKED*') {
    $status | ConvertTo-Json -Depth 8 -Compress
    exit 0
}
if ($action.decision -notin @('LAUNCH_SMOKE', 'LAUNCH_EPOCH1', 'RESUME_EPOCH1')) {
    throw 'Coordinator returned no admissible action'
}
if ($action.automatic_power_limit_change -ne $false -or $action.physical_power_limit_w_required -ne 160 -or $action.maximum_power_draw_w -ne 160) {
    throw 'Coordinator safety policy differs'
}
$manifestWindows = Convert-Gx1WslPath -LinuxPath $action.invocation_manifest.path
if ((Get-FileHash -LiteralPath $manifestWindows -Algorithm SHA256).Hash.ToLowerInvariant() -cne $action.invocation_manifest.sha256) {
    throw 'Invocation manifest changed after full launch gate'
}
$manifest = Get-Content -LiteralPath $manifestWindows -Raw -Encoding UTF8 | ConvertFrom-Json
$pointerWindows = Convert-Gx1WslPath -LinuxPath $manifest.pointer_path
$pointerBefore = (Get-FileHash -LiteralPath $pointerWindows -Algorithm SHA256).Hash.ToLowerInvariant()
$runtimeLinux = Split-Path -Parent (Split-Path -Parent $manifest.progress_json)
$runtimeWindows = Convert-Gx1WslPath -LinuxPath $runtimeLinux
$telemetry = Join-Path $runtimeWindows ('telemetry-invocation-{0:d4}.jsonl' -f $action.invocation_number)
$statusJson = Join-Path $runtimeWindows 'STATUS.json'
if (Test-Path -LiteralPath $telemetry) { throw 'Telemetry output already exists' }
$activeMarker = Join-Path $runtimeWindows 'ACTIVE_INVOCATION.json'
if (Test-Path -LiteralPath $activeMarker) { throw 'Unresolved prior invocation marker blocks launch' }
$activeTemporary = $activeMarker + '.tmp'
$activePayload = [ordered]@{ stage = [string]$action.stage; invocation_number = [int]$action.invocation_number }
[IO.File]::WriteAllText($activeTemporary, (($activePayload | ConvertTo-Json -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
[IO.File]::Move($activeTemporary, $activeMarker)
$wslArguments = @('-d', $Distro, '-u', $LinuxUser, '--cd', $SourceRepo, '--') + @($manifest.launcher_command)
$trainer = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArguments -PassThru -NoNewWindow
$telemetryScript = Join-Path $WindowsSourceRepo 'scripts/windows/GX1-LifecycleV2PilotTelemetry.ps1'
$telemetryProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $telemetryScript,
    '-TrainerProcessId', [string]$trainer.Id, '-Stage', [string]$action.stage,
    '-InvocationNumber', [string]$action.invocation_number, '-ExpectedGpuUuid', [string]$gpu.uuid,
    '-TelemetryJsonl', $telemetry, '-StatusJson', $statusJson,
    '-ProgressJson', (Convert-Gx1WslPath -LinuxPath $manifest.progress_json)
) -PassThru -NoNewWindow
$trainer.WaitForExit()
$telemetryProcess.WaitForExit()
$outcome = 'FAILED'
if ($trainer.ExitCode -eq 0) {
    if ($action.stage -ceq 'smoke') {
        $outcome = 'COMPLETE'
    }
    else {
        $progressWindows = Convert-Gx1WslPath -LinuxPath $manifest.progress_json
        $progress = Get-Content -LiteralPath $progressWindows -Raw -Encoding UTF8 | ConvertFrom-Json
        $outcome = if ($progress.completed_units -eq $progress.total_units) { 'COMPLETE' } else { 'RESUMABLE' }
    }
}
$telemetryLinux = @(& wsl.exe -d $Distro -u $LinuxUser -- wslpath -u $telemetry)
$receipt = Invoke-Gx1PilotJson -Arguments @(
    'record', '--plan-json', $PlanJson, '--plan-sha256', $PlanSha256,
    '--stage', [string]$action.stage, '--invocation-number', [string]$action.invocation_number,
    '--boot-utc', $bootUtc, '--outcome', $outcome,
    '--pointer-before-sha256', $pointerBefore, '--telemetry-jsonl', $telemetryLinux[0]
)
if ($outcome -ne 'COMPLETE' -or $action.stage -ceq 'smoke') {
    & shutdown.exe /r /t 30 /d p:0:0 /c 'GX1 lifecycle-v2 bounded invocation completed; reboot required'
}
$receipt | ConvertTo-Json -Depth 8 -Compress
