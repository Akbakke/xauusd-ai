param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanFileSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][string]$WindowsSourceRepo,
    [Parameter(Mandatory = $true)][string]$WindowsTaskUser,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$TaskName = 'GX1RandomAccessCampaignV2',
    [string]$CampaignControlRepo = $SourceRepo
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($WindowsTaskUser -ieq 'SYSTEM' -or $WindowsTaskUser -match '^NT AUTHORITY\\') {
    throw 'Campaign task must run as the explicit Windows account that owns the WSL distribution'
}
$controller = Join-Path $WindowsSourceRepo 'scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1'
$observer = Join-Path $WindowsSourceRepo 'scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1'
foreach ($path in @($controller, $observer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Campaign source unavailable: $path" }
}
$controllerSha256 = (Get-FileHash -LiteralPath $controller -Algorithm SHA256).Hash.ToLowerInvariant()
$legacyWslTask = Get-ScheduledTask -TaskName 'WSL SSH Bootstrap' -ErrorAction SilentlyContinue
if ($null -ne $legacyWslTask) {
    if ([string]$legacyWslTask.State -ceq 'Running') {
        Stop-ScheduledTask -TaskName 'WSL SSH Bootstrap' -ErrorAction Stop
    }
    Disable-ScheduledTask -TaskName 'WSL SSH Bootstrap' -ErrorAction Stop | Out-Null
    $legacyWslTask = Get-ScheduledTask -TaskName 'WSL SSH Bootstrap' -ErrorAction Stop
    if ([string]$legacyWslTask.State -cne 'Disabled') {
        throw 'Legacy WSL SSH Bootstrap task could not be disabled'
    }
}
$arguments = @(
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $controller + '"'),
    '-PlanJson', ('"' + $PlanJson + '"'),
    '-PlanFileSha256', $PlanFileSha256,
    '-SourceRepo', ('"' + $SourceRepo + '"'),
    '-CampaignControlRepo', ('"' + $CampaignControlRepo + '"'),
    '-WindowsSourceRepo', ('"' + $WindowsSourceRepo + '"'),
    '-ExpectedControllerSha256', $controllerSha256,
    '-Distro', ('"' + $Distro + '"'),
    '-LinuxUser', ('"' + $LinuxUser + '"')
) -join ' '
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = 'PT60S'
$principal = New-ScheduledTaskPrincipal -UserId $WindowsTaskUser -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$registeredDelaySeconds = [Xml.XmlConvert]::ToTimeSpan([string]$registered.Triggers[0].Delay).TotalSeconds
$legacyWslTask = Get-ScheduledTask -TaskName 'WSL SSH Bootstrap' -ErrorAction SilentlyContinue
if ($registered.Principal.UserId -cne $WindowsTaskUser -or
    $registered.Principal.LogonType -cne 'S4U' -or
    @($registered.Triggers).Count -ne 1 -or
    $registered.Triggers[0].CimClass.CimClassName -cne 'MSFT_TaskBootTrigger' -or
    $registeredDelaySeconds -ne 60 -or
    @($registered.Actions).Count -ne 1 -or
    [string]$registered.Actions[0].Arguments -cne $arguments -or
    [int]$registered.Settings.RestartCount -ne 0 -or
    ($null -ne $legacyWslTask -and [string]$legacyWslTask.State -cne 'Disabled')) {
    throw 'Campaign task single-owner, principal, action, or delayed-trigger verification failed'
}
[ordered]@{
    schema_version = 'gx1_random_access_campaign_v2_task_install_receipt_v1'
    decision = 'PASS_INSTALLED_REBOOT_REQUIRED'
    task_name = $TaskName
    windows_task_user = $WindowsTaskUser
    logon_type = [string]$registered.Principal.LogonType
    source_repo = $SourceRepo
    campaign_control_repo = $CampaignControlRepo
    controller_path = $controller
    controller_file_sha256 = $controllerSha256
    legacy_wsl_ssh_bootstrap_disabled = ($null -eq $legacyWslTask -or [string]$legacyWslTask.State -ceq 'Disabled')
    boot_trigger_delay = [string]$registered.Triggers[0].Delay
    automatic_restart_count = [int]$registered.Settings.RestartCount
    observer_path = $observer
    observer_file_sha256 = (Get-FileHash -LiteralPath $observer -Algorithm SHA256).Hash.ToLowerInvariant()
    plan_file_sha256 = $PlanFileSha256
    first_launch_requires_next_windows_boot = $true
} | ConvertTo-Json -Compress
