param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanFileSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][string]$WindowsSourceRepo,
    [Parameter(Mandatory = $true)][string]$WindowsTaskUser,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$TaskName = 'GX1RandomAccessCampaignV2'
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
$arguments = @(
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $controller + '"'),
    '-PlanJson', ('"' + $PlanJson + '"'),
    '-PlanFileSha256', $PlanFileSha256,
    '-SourceRepo', ('"' + $SourceRepo + '"'),
    '-WindowsSourceRepo', ('"' + $WindowsSourceRepo + '"'),
    '-Distro', ('"' + $Distro + '"'),
    '-LinuxUser', ('"' + $LinuxUser + '"')
) -join ' '
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId $WindowsTaskUser -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
if ($registered.Principal.UserId -cne $WindowsTaskUser -or $registered.Principal.LogonType -cne 'S4U') {
    throw 'Campaign task principal verification failed'
}
[ordered]@{
    schema_version = 'gx1_random_access_campaign_v2_task_install_receipt_v1'
    decision = 'PASS_INSTALLED_REBOOT_REQUIRED'
    task_name = $TaskName
    windows_task_user = $WindowsTaskUser
    logon_type = [string]$registered.Principal.LogonType
    controller_path = $controller
    controller_file_sha256 = (Get-FileHash -LiteralPath $controller -Algorithm SHA256).Hash.ToLowerInvariant()
    observer_path = $observer
    observer_file_sha256 = (Get-FileHash -LiteralPath $observer -Algorithm SHA256).Hash.ToLowerInvariant()
    plan_file_sha256 = $PlanFileSha256
    first_launch_requires_next_windows_boot = $true
} | ConvertTo-Json -Compress
