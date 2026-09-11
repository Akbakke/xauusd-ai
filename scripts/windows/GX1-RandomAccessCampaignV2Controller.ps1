param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanFileSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][Alias('WindowsSourceRepo')][string]$WindowsControllerSourceRoot,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$Python = '.venv/bin/python'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Invoke-Gx1Json {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $lines = @(& wsl.exe -d $Distro -u $LinuxUser --cd $SourceRepo -- $Python -m gx1.scripts.local_random_access_campaign_v2 @Arguments)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1) { throw 'Random-access campaign command failed' }
    $value = $lines[0] | ConvertFrom-Json
    if ($value.ok -ne $true) { throw 'Random-access campaign command returned non-PASS' }
    return $value
}
function Convert-Gx1WslPath {
    param([Parameter(Mandatory = $true)][string]$LinuxPath)
    $value = @(& wsl.exe -d $Distro -u $LinuxUser -- wslpath -w $LinuxPath)
    if ($LASTEXITCODE -ne 0 -or $value.Count -ne 1) { throw 'WSL path conversion failed' }
    return $value[0]
}
function Write-Gx1BootIdentity {
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $bootId = (Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters' -Name BootId -ErrorAction Stop).BootId
    $bootUtc = $operatingSystem.LastBootUpTime.ToUniversalTime().ToString('o')
    $computer = [Environment]::MachineName
    $unsigned = '{"boot_id":' + [string]$bootId + ',"computer_name":"' + $computer + '","last_boot_utc":"' + $bootUtc + '","schema_version":"gx1_windows_boot_identity_v1"}' + "`n"
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($unsigned)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $digestBytes = $sha.ComputeHash($bytes) } finally { $sha.Dispose() }
    $digest = ([BitConverter]::ToString($digestBytes) -replace '-', '').ToLowerInvariant()
    $root = Join-Path $env:ProgramData 'GX1\RandomAccessCampaignV2'
    New-Item -ItemType Directory -Path $root -Force | Out-Null
    $path = Join-Path $root 'CURRENT_BOOT.json'
    $value = [ordered]@{
        schema_version = 'gx1_windows_boot_identity_v1'
        computer_name = $computer
        last_boot_utc = $bootUtc
        boot_id = [int64]$bootId
        identity_sha256 = $digest
    }
    $temporary = $path + '.tmp.' + [guid]::NewGuid().ToString('N')
    $backup = $path + '.backup.' + [guid]::NewGuid().ToString('N')
    try {
        [IO.File]::WriteAllText($temporary, (($value | ConvertTo-Json -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $path) {
            [IO.File]::Replace($temporary, $path, $backup)
            Remove-Item -LiteralPath $backup -Force
        }
        else {
            [IO.File]::Move($temporary, $path)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
        if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Force }
    }
    # wsl.exe consumes one layer of backslash escaping before wslpath sees the
    # argument. Keep the filesystem path untouched and escape only this argv.
    $escapedPathForWsl = $path.Replace('\', '\\')
    $linux = @(& wsl.exe -d $Distro -u $LinuxUser -- wslpath -u $escapedPathForWsl)
    if ($LASTEXITCODE -ne 0 -or $linux.Count -ne 1) { throw 'Boot identity path conversion failed' }
    return [pscustomobject]@{ Windows = $path; Linux = $linux[0]; Payload = $value }
}
function Request-Gx1PhysicalReboot {
    param([object]$Boot)
    $prepared = Invoke-Gx1Json -Arguments @(
        'prepare-reboot', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
        '--boot-json', $Boot.Linux
    )
    & shutdown.exe /r /t 60 /d p:0:0 /c 'GX1 random-access campaign requires a fresh physical Windows boot'
    $shutdownExit = $LASTEXITCODE
    if ($shutdownExit -ne 0) { throw 'shutdown.exe rejected physical reboot request' }
    [void](Invoke-Gx1Json -Arguments @(
        'confirm-reboot', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
        '--request-nonce', [string]$prepared.intent.request_nonce,
        '--shutdown-exit-code', [string]$shutdownExit
    ))
}
$boot = Write-Gx1BootIdentity
$status = Invoke-Gx1Json -Arguments @(
    'inspect', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--boot-json', $boot.Linux
)
$controllerSourceRoot = [IO.Path]::GetFullPath($WindowsControllerSourceRoot).TrimEnd('\')
if ($controllerSourceRoot -notmatch '^[Cc]:\\' -or
    -not (Test-Path -LiteralPath $controllerSourceRoot -PathType Container) -or
    ((Get-Item -LiteralPath $controllerSourceRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Windows controller source root must be an existing local C: directory without a reparse point'
}
$controllerBinding = $status.controller_sources.controller
$observerBinding = $status.controller_sources.observer
$expectedControllerSource = Join-Path $controllerSourceRoot 'scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1'
$observerSource = Join-Path $controllerSourceRoot 'scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1'
foreach ($source in @($expectedControllerSource, $observerSource)) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or
        ((Get-Item -LiteralPath $source -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Windows controller staging source is unavailable or is a reparse point'
    }
}
if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
        [IO.Path]::GetFullPath($PSCommandPath),
        [IO.Path]::GetFullPath($expectedControllerSource))) {
    throw 'Running campaign controller is outside the explicit staging root'
}
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$controllerBinding.sha256 -or
    (Get-FileHash -LiteralPath $observerSource -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$observerBinding.sha256) {
    throw 'Campaign controller or observer differs from immutable plan binding'
}
if ($status.policy.physical_power_limit_w -ne 160 -or
    $status.policy.maximum_actual_power_draw_w -ne 170 -or
    $status.policy.signed_local_telemetry_seconds -ne 1 -or
    $status.policy.human_status_seconds -ne 900) {
    throw 'Campaign safety policy differs'
}
if ($status.action.decision -ceq 'REBOOT_REQUIRED') {
    Request-Gx1PhysicalReboot -Boot $boot
    exit 0
}
if ($status.action.decision -ceq 'COMPLETE' -or $status.action.decision -like 'BLOCKED*') {
    $status | ConvertTo-Json -Depth 16 -Compress
    exit 0
}
if ($status.action.decision -cne 'LAUNCH') { throw 'Campaign returned no admissible action' }
$begin = Invoke-Gx1Json -Arguments @(
    'begin', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--boot-json', $boot.Linux
)
$invocation = $begin.invocation
$argv = @($invocation.launcher_argv | ForEach-Object { [string]$_ })
$progressWindows = Convert-Gx1WslPath -LinuxPath ([string]$invocation.progress_path)
$guardWindows = Convert-Gx1WslPath -LinuxPath ([string]$invocation.guard_log_path)
$runtimeWindows = Split-Path -Parent $progressWindows
$statusJson = Join-Path $runtimeWindows ('STATUS-{0}.json' -f $invocation.invocation_id)
$humanJsonl = Join-Path $runtimeWindows 'HUMAN_STATUS.jsonl'
if (Test-Path -LiteralPath $guardWindows) { throw 'Guard log already exists before invocation' }
$trainerArguments = @('-d', $Distro, '-u', $LinuxUser, '--cd', $SourceRepo, '--') + $argv
$priorPlanSha = $env:GX1_CAMPAIGN_PLAN_SHA256
$priorPlanPath = $env:GX1_CAMPAIGN_PLAN_PATH
$priorPlanFileSha = $env:GX1_CAMPAIGN_PLAN_FILE_SHA256
$priorInvocationSha = $env:GX1_CAMPAIGN_INVOCATION_SHA256
$priorGuardLogPath = $env:GX1_CAMPAIGN_GUARD_LOG_PATH
$priorWslEnv = $env:WSLENV
$campaignWslEnv = 'GX1_CAMPAIGN_PLAN_SHA256:GX1_CAMPAIGN_PLAN_PATH:GX1_CAMPAIGN_PLAN_FILE_SHA256:GX1_CAMPAIGN_INVOCATION_SHA256:GX1_CAMPAIGN_GUARD_LOG_PATH'
try {
    $env:GX1_CAMPAIGN_PLAN_SHA256 = [string]$status.plan_sha256
    if (-not $PlanJson.StartsWith('/')) { throw 'Campaign plan must be an absolute WSL path' }
    $env:GX1_CAMPAIGN_PLAN_PATH = $PlanJson
    $env:GX1_CAMPAIGN_PLAN_FILE_SHA256 = $PlanFileSha256
    $env:GX1_CAMPAIGN_INVOCATION_SHA256 = [string]$invocation.invocation_sha256
    $env:GX1_CAMPAIGN_GUARD_LOG_PATH = [string]$invocation.guard_log_path
    $env:WSLENV = if ([string]::IsNullOrWhiteSpace($priorWslEnv)) {
        $campaignWslEnv
    }
    else {
        "$priorWslEnv`:$campaignWslEnv"
    }
    $trainer = Start-Process -FilePath 'wsl.exe' -ArgumentList $trainerArguments -PassThru -NoNewWindow
}
finally {
    $env:GX1_CAMPAIGN_PLAN_SHA256 = $priorPlanSha
    $env:GX1_CAMPAIGN_PLAN_PATH = $priorPlanPath
    $env:GX1_CAMPAIGN_PLAN_FILE_SHA256 = $priorPlanFileSha
    $env:GX1_CAMPAIGN_INVOCATION_SHA256 = $priorInvocationSha
    $env:GX1_CAMPAIGN_GUARD_LOG_PATH = $priorGuardLogPath
    $env:WSLENV = $priorWslEnv
}
$observerScript = Join-Path $controllerSourceRoot 'scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1'
$observer = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $observerScript,
    '-TrainerProcessId', [string]$trainer.Id,
    '-PlanSha256', [string]$status.plan_sha256,
    '-InvocationSha256', [string]$invocation.invocation_sha256,
    '-ProgressJson', $progressWindows,
    '-StatusJson', $statusJson,
    '-HumanStatusJsonl', $humanJsonl
) -PassThru -NoNewWindow
$trainer.WaitForExit()
$observer.WaitForExit()
$outcome = if ($trainer.ExitCode -eq 0 -and $observer.ExitCode -eq 0) {
    [string]$invocation.expected_success_outcome
}
else {
    'FAILED'
}
$recorded = Invoke-Gx1Json -Arguments @(
    'record', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--trainer-guard-exit-code', [string]$trainer.ExitCode,
    '--progress-observer-exit-code', [string]$observer.ExitCode,
    '--outcome', $outcome
)
if ($outcome -eq 'FAILED') {
    $recorded | ConvertTo-Json -Depth 16 -Compress
    exit 2
}
$after = Invoke-Gx1Json -Arguments @(
    'inspect', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--boot-json', $boot.Linux
)
if ($after.action.decision -eq 'COMPLETE') {
    $recorded | ConvertTo-Json -Depth 16 -Compress
    exit 0
}
Request-Gx1PhysicalReboot -Boot $boot
$recorded | ConvertTo-Json -Depth 16 -Compress
