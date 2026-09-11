param(
    [Parameter(Mandatory = $true)][int]$TrainerProcessId,
    [Parameter(Mandatory = $true)][string]$PlanSha256,
    [Parameter(Mandatory = $true)][string]$InvocationSha256,
    [Parameter(Mandatory = $true)][string]$ProgressJson,
    [Parameter(Mandatory = $true)][string]$StatusJson,
    [Parameter(Mandatory = $true)][string]$HumanStatusJsonl
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Progress-only observer. GPU safety and one-second signed hardware telemetry are
# owned exclusively by gx1_capped_run.sh -> gx1_guarded_trainer_exec.sh.
function Write-Gx1AtomicJson {
    param([string]$Path, [object]$Value)
    $temporary = $Path + '.tmp.' + [guid]::NewGuid().ToString('N')
    $backup = $Path + '.backup.' + [guid]::NewGuid().ToString('N')
    $payload = ($Value | ConvertTo-Json -Depth 12 -Compress) + [Environment]::NewLine
    try {
        [IO.File]::WriteAllText($temporary, $payload, [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $Path) {
            [IO.File]::Replace($temporary, $Path, $backup)
            Remove-Item -LiteralPath $backup -Force
        }
        else {
            [IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
        if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Force }
    }
}
function Read-Gx1Progress {
    if (-not (Test-Path -LiteralPath $ProgressJson -PathType Leaf)) { return $null }
    $value = Get-Content -LiteralPath $ProgressJson -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($value.schema_version -cne 'gx1_local_random_access_progress_v2' -or
        $value.plan_sha256 -cne $PlanSha256 -or
        $value.invocation_sha256 -cne $InvocationSha256) {
        throw 'Random-access progress identity invalid'
    }
    return $value
}
$lastHumanUtc = $null
$lastPhase = $null
$lastCompleted = $null
$lastOutcome = $null
while ($null -ne (Get-Process -Id $TrainerProcessId -ErrorAction SilentlyContinue)) {
    $now = [datetime]::UtcNow
    $progress = Read-Gx1Progress
    $phase = if ($null -eq $progress) { $null } else { [string]$progress.phase }
    $completed = if ($null -eq $progress) { $null } else { [int64]$progress.completed_units }
    $total = if ($null -eq $progress) { $null } else { [int64]$progress.total_units }
    $outcome = if ($null -eq $progress) { $null } else { $progress.outcome }
    if ($null -ne $progress -and ($completed -lt 0 -or $total -lt 1 -or $completed -gt $total)) {
        throw 'Random-access progress range invalid'
    }
    $meaningful = (
        $null -eq $lastHumanUtc -or
        $phase -cne $lastPhase -or
        $outcome -cne $lastOutcome -or
        ($null -ne $completed -and $completed -eq $total)
    )
    $humanDue = $meaningful -or ($now - $lastHumanUtc).TotalSeconds -ge 900
    $status = [ordered]@{
        schema_version = 'gx1_local_random_access_status_v2'
        observed_utc = $now.ToString('o')
        plan_sha256 = $PlanSha256
        invocation_sha256 = $InvocationSha256
        phase = $phase
        completed_units = $completed
        total_units = $total
        outcome = $outcome
        meaningful_event = $meaningful
        human_status_due = $humanDue
        local_progress_sample_seconds = 1
        signed_hardware_telemetry_owner = 'gx1_guarded_trainer_exec.sh'
        signed_hardware_telemetry_seconds = 1
        human_status_seconds = 900
        codex_polling_required = $false
    }
    Write-Gx1AtomicJson -Path $StatusJson -Value $status
    if ($humanDue) {
        [IO.File]::AppendAllText(
            $HumanStatusJsonl,
            (($status | ConvertTo-Json -Depth 12 -Compress) + [Environment]::NewLine),
            [Text.UTF8Encoding]::new($false)
        )
        $lastHumanUtc = $now
    }
    $lastPhase = $phase
    $lastCompleted = $completed
    $lastOutcome = $outcome
    Start-Sleep -Seconds 1
}
$terminal = Read-Gx1Progress
if ($null -eq $terminal -or $terminal.terminal -ne $true) {
    throw 'Trainer exited without terminal random-access progress receipt'
}
