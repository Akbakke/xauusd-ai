param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][string]$WindowsSourceRepo,
    [string]$TaskName = 'GX1LifecycleV2PilotResume'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$controller = Join-Path $WindowsSourceRepo 'scripts/windows/GX1-LifecycleV2PilotController.ps1'
$arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $controller +
    '" -PlanJson "' + $PlanJson + '" -PlanSha256 "' + $PlanSha256 +
    '" -SourceRepo "' + $SourceRepo + '" -WindowsSourceRepo "' + $WindowsSourceRepo + '"'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force
