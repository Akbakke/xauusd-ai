param(
    [Parameter(Mandatory = $true)][int]$TrainerProcessId,
    [Parameter(Mandatory = $true)][string]$Stage,
    [Parameter(Mandatory = $true)][int]$InvocationNumber,
    [Parameter(Mandatory = $true)][string]$ExpectedGpuUuid,
    [Parameter(Mandatory = $true)][string]$TelemetryJsonl,
    [Parameter(Mandatory = $true)][string]$StatusJson,
    [string]$ProgressJson = '',
    [string]$NvidiaSmiPath = 'C:\Windows\System32\nvidia-smi.exe'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Observation only: this source must never invoke a power-setting command or keeper.
$lastCompleted = $null
$lastObserved = $null
$lastPhase = $null
$lastHumanUtc = $null
while ($null -ne (Get-Process -Id $TrainerProcessId -ErrorAction SilentlyContinue)) {
    $now = [datetime]::UtcNow
    $raw = @(& $NvidiaSmiPath --query-gpu=uuid,power.limit,power.draw,temperature.gpu,temperature.memory,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $raw.Count -ne 1) { throw 'GPU telemetry unavailable' }
    $v = @($raw[0].Split(',') | ForEach-Object { $_.Trim() })
    if ($v.Count -ne 7 -or $v[0] -cne $ExpectedGpuUuid) { throw 'GPU telemetry identity invalid' }
    $limit = [double]::Parse($v[1], [Globalization.CultureInfo]::InvariantCulture)
    $draw = [double]::Parse($v[2], [Globalization.CultureInfo]::InvariantCulture)
    $phase = $null; $completed = $null; $total = $null; $eta = $null
    if (-not [string]::IsNullOrWhiteSpace($ProgressJson) -and (Test-Path -LiteralPath $ProgressJson -PathType Leaf)) {
        $progress = Get-Content -LiteralPath $ProgressJson -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($progress.schema_version -cne 'gx1_local_lifecycle_v2_progress_v1' -or
            $progress.completed_units -lt 0 -or $progress.total_units -lt 1 -or
            $progress.completed_units -gt $progress.total_units) { throw 'Progress JSON invalid' }
        $phase = [string]$progress.phase
        $completed = [int64]$progress.completed_units
        $total = [int64]$progress.total_units
        if ($null -ne $lastCompleted -and $completed -gt $lastCompleted -and $phase -ceq $lastPhase) {
            $seconds = ($now - $lastObserved).TotalSeconds
            $eta = [double](($total - $completed) * $seconds / ($completed - $lastCompleted))
        }
    }
    $decision = if ($limit -gt 160.0 -or $draw -gt 160.0) { 'KILL_AND_BLOCK' } else { 'PASS' }
    $sample = [ordered]@{
        schema_version = 'gx1_local_lifecycle_v2_pilot_telemetry_v1'
        observed_utc = $now.ToString('o'); stage = $Stage
        invocation_number = $InvocationNumber; gpu_uuid = $v[0]
        power_limit_w = $limit; power_draw_w = $draw
        core_temp_c = [double]$v[3]; memory_temp_c = [double]$v[4]
        memory_used_mib = [int]$v[5]; utilization_percent = [int]$v[6]
        phase = $phase; completed_units = $completed; total_units = $total
        eta_seconds = $eta; safety_decision = $decision
    }
    [IO.File]::AppendAllText($TelemetryJsonl, (($sample | ConvertTo-Json -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    $meaningful = ($null -eq $lastObserved -or $phase -cne $lastPhase -or $decision -cne 'PASS' -or ($null -ne $completed -and $completed -eq $total))
    $humanDue = ($meaningful -or $null -eq $lastHumanUtc -or ($now - $lastHumanUtc).TotalSeconds -ge 900)
    if ($humanDue) { $lastHumanUtc = $now }
    $status = [ordered]@{
        schema_version = 'gx1_local_lifecycle_v2_pilot_status_v1'
        observed_utc = $now.ToString('o'); stage = $Stage; invocation_number = $InvocationNumber
        phase = $phase; completed_units = $completed; total_units = $total; eta_seconds = $eta
        power_limit_w = $limit; power_draw_w = $draw; safety_decision = $decision
        meaningful_event = $meaningful; human_status_due = $humanDue
        last_human_status_utc = if ($null -eq $lastHumanUtc) { $null } else { $lastHumanUtc.ToString('o') }
        human_status_cadence_seconds = 900; local_sample_seconds = 1
        codex_polling_required = $false
    }
    $temporary = $StatusJson + '.tmp.' + [guid]::NewGuid().ToString('N')
    [IO.File]::WriteAllText($temporary, (($status | ConvertTo-Json -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $StatusJson) { [IO.File]::Replace($temporary, $StatusJson, $null) } else { [IO.File]::Move($temporary, $StatusJson) }
    if ($decision -cne 'PASS') {
        Stop-Process -Id $TrainerProcessId -Force -ErrorAction SilentlyContinue
        break
    }
    $lastCompleted = $completed; $lastObserved = $now; $lastPhase = $phase
    Start-Sleep -Seconds 1
}
