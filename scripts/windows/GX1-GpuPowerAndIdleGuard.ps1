[CmdletBinding()]
param(
    [switch]$Once,
    [switch]$PolicySelfTest,
    [string]$BenchmarkScopePath = '',
    [string]$BenchmarkScopeSha256 = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$configPath = Join-Path $root 'GX1-GpuPowerLimit.config.json'
$logPath = Join-Path $root 'GX1-GpuPowerLimit.log'
$statePath = Join-Path $root 'GX1-GpuIdleGuard.state.json'
$blockerPath = Join-Path $root 'GX1-GpuIdleGuard.block.json'

function Write-Gx1GuardLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "$(Get-Date -Format o) $Message"
    [System.IO.File]::AppendAllText(
        $logPath,
        "$line`r`n",
        [System.Text.UTF8Encoding]::new($false)
    )
}

function ConvertTo-Gx1FiniteDouble {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $parsed = 0.0
    if (-not [double]::TryParse(
        $Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsed
    ) -or [double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) {
        throw "$Name is not finite: '$Value'"
    }
    return $parsed
}

function Get-Gx1GpuSample {
    param([Parameter(Mandatory = $true)][psobject]$Config)

    if (-not (Test-Path -LiteralPath ([string]$Config.native_smi) -PathType Leaf)) {
        throw 'nvidia-smi.exe is unavailable'
    }
    $query = 'name,uuid,pstate,temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu'
    $raw = @(& ([string]$Config.native_smi) -i ([string]$Config.gpu_index) "--query-gpu=$query" '--format=csv,noheader,nounits' 2>&1)
    if ($LASTEXITCODE -ne 0 -or $raw.Count -ne 1) {
        throw "nvidia-smi telemetry query failed: $(($raw | Out-String).Trim())"
    }
    $fields = @($raw[0].ToString().Split(',') | ForEach-Object { $_.Trim() })
    if ($fields.Count -ne 8 -or
        $fields[0] -ne [string]$Config.expected_gpu_name -or
        $fields[1] -ne [string]$Config.expected_gpu_uuid -or
        $fields[2] -notmatch '^P([0-9]|1[0-2])$') {
        throw "GPU identity or pstate mismatch: '$($raw[0])'"
    }
    $memoryUsed = 0
    $utilization = 0
    if (-not [int]::TryParse($fields[6], [ref]$memoryUsed) -or $memoryUsed -lt 0 -or
        -not [int]::TryParse($fields[7], [ref]$utilization) -or
        $utilization -lt 0 -or $utilization -gt 100) {
        throw "GPU memory/utilization telemetry is invalid: '$($raw[0])'"
    }
    return [pscustomobject]@{
        observed_utc = [datetime]::UtcNow.ToString('o')
        gpu_name = $fields[0]
        gpu_uuid = $fields[1]
        pstate = $fields[2]
        core_temp_c = ConvertTo-Gx1FiniteDouble -Name 'core_temp_c' -Value $fields[3]
        power_draw_w = ConvertTo-Gx1FiniteDouble -Name 'power_draw_w' -Value $fields[4]
        power_limit_w = ConvertTo-Gx1FiniteDouble -Name 'power_limit_w' -Value $fields[5]
        memory_used_mib = $memoryUsed
        utilization_percent = $utilization
    }
}

function Test-Gx1HighIdleSample {
    param(
        [Parameter(Mandatory = $true)][psobject]$Sample,
        [Parameter(Mandatory = $true)][psobject]$Config
    )

    return [bool](
        $Sample.pstate -match '^P[0-2]$' -and
        $Sample.power_draw_w -gt [double]$Config.idle_power_threshold_w -and
        $Sample.memory_used_mib -le [int]$Config.idle_memory_max_mib -and
        $Sample.utilization_percent -le [int]$Config.idle_utilization_max_percent
    )
}

function Test-Gx1NormalIdleSample {
    param(
        [Parameter(Mandatory = $true)][psobject]$Sample,
        [Parameter(Mandatory = $true)][psobject]$Config
    )

    return [bool](
        $Sample.power_draw_w -le [double]$Config.idle_power_threshold_w -and
        $Sample.memory_used_mib -le [int]$Config.idle_memory_max_mib -and
        $Sample.utilization_percent -le [int]$Config.idle_utilization_max_percent
    )
}

function Set-Gx1PowerLimit {
    param([Parameter(Mandatory = $true)][psobject]$Config)

    $setOutput = @(& ([string]$Config.native_smi) -i ([string]$Config.gpu_index) -pl ([string]$Config.power_limit_w) 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi -pl failed: $(($setOutput | Out-String).Trim())"
    }
    $sample = Get-Gx1GpuSample -Config $Config
    if ($sample.power_limit_w -gt [double]$Config.power_limit_w) {
        throw "power-limit verification mismatch: $($sample.power_limit_w)W"
    }
    return $sample
}

function Stop-Gx1TelemetryBridge {
    param([Parameter(Mandatory = $true)][psobject]$Config)

    $task = Get-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction SilentlyContinue
    if ($null -ne $task -and $task.State -eq 'Running') {
        Stop-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction Stop
        foreach ($attempt in 1..20) {
            Start-Sleep -Milliseconds 250
            $task = Get-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction SilentlyContinue
            if ($null -eq $task -or $task.State -ne 'Running') {
                return
            }
        }
        throw "telemetry task did not stop: $($Config.telemetry_task_name)"
    }
}

function Start-Gx1TelemetryBridge {
    param([Parameter(Mandatory = $true)][psobject]$Config)

    $task = Get-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction Stop
    if ($task.State -eq 'Running') {
        Stop-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction Stop
        Start-Sleep -Seconds 1
    }
    Start-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction Stop
    foreach ($attempt in 1..20) {
        Start-Sleep -Milliseconds 250
        $task = Get-ScheduledTask -TaskName ([string]$Config.telemetry_task_name) -ErrorAction SilentlyContinue
        if ($null -ne $task -and $task.State -eq 'Running') {
            return
        }
    }
    throw "telemetry task did not start: $($Config.telemetry_task_name)"
}

function Get-Gx1RecoveryHistory {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        return @()
    }
    try {
        $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($state.schema_version -ne 'gx1_gpu_idle_guard_state_v1') {
            throw 'state schema mismatch'
        }
        return @($state.recovery_attempts_utc | ForEach-Object {
            [datetime]::Parse($_, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
        })
    }
    catch {
        Write-Gx1GuardLog "STATE_INVALID message=$($_.Exception.Message)"
        throw
    }
}

function Save-Gx1RecoveryHistory {
    param([Parameter(Mandatory = $true)][datetime[]]$History)

    $state = [ordered]@{
        schema_version = 'gx1_gpu_idle_guard_state_v1'
        updated_utc = [datetime]::UtcNow.ToString('o')
        recovery_attempts_utc = @($History | ForEach-Object { $_.ToUniversalTime().ToString('o') })
    }
    $temporaryPath = "$statePath.$PID.tmp"
    [System.IO.File]::WriteAllText(
        $temporaryPath,
        ($state | ConvertTo-Json -Compress),
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporaryPath -Destination $statePath -Force
}

function Write-Gx1Blocker {
    param(
        [Parameter(Mandatory = $true)][psobject]$Sample,
        [Parameter(Mandatory = $true)][string]$Reason
    )

    $blocker = [ordered]@{
        schema_version = 'gx1_gpu_idle_guard_block_v1'
        decision = 'BLOCK_CUDA'
        reason = $Reason
        observed_utc = [datetime]::UtcNow.ToString('o')
        sample = $Sample
    }
    [System.IO.File]::WriteAllText(
        $blockerPath,
        ($blocker | ConvertTo-Json -Depth 4 -Compress),
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Clear-Gx1Blocker {
    if (Test-Path -LiteralPath $blockerPath -PathType Leaf) {
        Remove-Item -LiteralPath $blockerPath -Force
    }
}

function Invoke-Gx1GpuRecovery {
    param(
        [Parameter(Mandatory = $true)][psobject]$Config,
        [Parameter(Mandatory = $true)][psobject]$BeforeSample
    )

    Write-Gx1Blocker -Sample $BeforeSample -Reason 'PERSISTENT_HIGH_IDLE_POWER_RECOVERY_ACTIVE'
    Stop-Gx1TelemetryBridge -Config $Config
    $pnpUtil = Join-Path $env:WINDIR 'System32\pnputil.exe'
    $restartOutput = @(& $pnpUtil '/restart-device' ([string]$Config.gpu_pnp_instance_id) 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "PnP GPU restart failed: $(($restartOutput | Out-String).Trim())"
    }
    $driverReady = $false
    foreach ($attempt in 1..([int]$Config.driver_ready_retry_count)) {
        Start-Sleep -Seconds ([int]$Config.driver_ready_retry_seconds)
        try {
            $sample = Get-Gx1GpuSample -Config $Config
            $driverReady = $true
            break
        }
        catch {
            if ($attempt -eq [int]$Config.driver_ready_retry_count) {
                throw
            }
        }
    }
    if (-not $driverReady) {
        throw 'Nvidia driver did not return after PnP restart'
    }
    Set-Gx1PowerLimit -Config $Config | Out-Null
    Start-Sleep -Seconds ([int]$Config.post_recovery_settle_seconds)
    $verificationSamples = @()
    foreach ($attempt in 1..([int]$Config.post_recovery_verification_samples)) {
        $sample = Get-Gx1GpuSample -Config $Config
        $verificationSamples += $sample
        if (-not (Test-Gx1NormalIdleSample -Sample $sample -Config $Config)) {
            throw "GPU failed post-recovery idle verification: pstate=$($sample.pstate) power_draw_w=$($sample.power_draw_w) memory_used_mib=$($sample.memory_used_mib) utilization_percent=$($sample.utilization_percent)"
        }
        if ($attempt -lt [int]$Config.post_recovery_verification_samples) {
            Start-Sleep -Seconds ([int]$Config.post_recovery_verification_interval_seconds)
        }
    }
    Start-Gx1TelemetryBridge -Config $Config
    Clear-Gx1Blocker
    return $verificationSamples[-1]
}

function Invoke-Gx1PolicySelfTest {
    $testConfig = [pscustomobject]@{
        idle_power_threshold_w = 60
        idle_memory_max_mib = 384
        idle_utilization_max_percent = 2
        idle_required_samples = 24
    }
    $normal = [pscustomobject]@{
        pstate = 'P8'
        power_draw_w = 25.0
        memory_used_mib = 72
        utilization_percent = 0
    }
    $stuck = [pscustomobject]@{
        pstate = 'P0'
        power_draw_w = 137.0
        memory_used_mib = 297
        utilization_percent = 0
    }
    $active = [pscustomobject]@{
        pstate = 'P0'
        power_draw_w = 158.0
        memory_used_mib = 9417
        utilization_percent = 80
    }
    $initialized = [pscustomobject]@{
        pstate = 'P0'
        power_draw_w = 95.0
        memory_used_mib = 545
        utilization_percent = 0
    }
    $smallWorkload = [pscustomobject]@{
        pstate = 'P0'
        power_draw_w = 95.0
        memory_used_mib = 300
        utilization_percent = 80
    }
    if (Test-Gx1HighIdleSample -Sample $normal -Config $testConfig) {
        throw 'normal idle was misclassified'
    }
    if (-not (Test-Gx1HighIdleSample -Sample $stuck -Config $testConfig)) {
        throw 'persistent high idle was not classified'
    }
    if (Test-Gx1HighIdleSample -Sample $active -Config $testConfig) {
        throw 'active training was misclassified'
    }
    if (Test-Gx1HighIdleSample -Sample $initialized -Config $testConfig) {
        throw 'initialized CUDA context was misclassified'
    }
    if (Test-Gx1HighIdleSample -Sample $smallWorkload -Config $testConfig) {
        throw 'small active CUDA workload was misclassified'
    }
    $counter = 0
    foreach ($sampleNumber in 1..([int]$testConfig.idle_required_samples - 1)) {
        if (Test-Gx1HighIdleSample -Sample $stuck -Config $testConfig) {
            $counter += 1
        }
        if ($counter -ge [int]$testConfig.idle_required_samples) {
            throw 'recovery threshold triggered early'
        }
    }
    if (Test-Gx1HighIdleSample -Sample $normal -Config $testConfig) {
        $counter += 1
    }
    else {
        $counter = 0
    }
    if ($counter -ne 0) {
        throw 'normal sample did not reset the sustained counter'
    }
    foreach ($sampleNumber in 1..([int]$testConfig.idle_required_samples)) {
        if (Test-Gx1HighIdleSample -Sample $stuck -Config $testConfig) {
            $counter += 1
        }
        if ($sampleNumber -lt [int]$testConfig.idle_required_samples -and
            $counter -ge [int]$testConfig.idle_required_samples) {
            throw 'recovery threshold triggered early after reset'
        }
    }
    if ($counter -ne [int]$testConfig.idle_required_samples) {
        throw 'recovery threshold did not trigger exactly'
    }
    [pscustomobject]@{
        schema_version = 'gx1_gpu_idle_guard_policy_self_test_v1'
        decision = 'PASS'
        required_samples = $counter
        normal_idle_rejected = $true
        active_training_rejected = $true
        initialized_cuda_rejected = $true
        small_active_workload_rejected = $true
        interruption_reset_verified = $true
        persistent_high_idle_detected = $true
    } | ConvertTo-Json -Compress
}

$benchmarkRequested = ($BenchmarkScopePath -ne '' -or $BenchmarkScopeSha256 -ne '')
if ($benchmarkRequested -and ($Once -or $PolicySelfTest -or $BenchmarkScopePath -eq '' -or $BenchmarkScopeSha256 -eq '')) {
    throw 'A benchmark requires both explicit operator scope arguments and continuous keeper mode'
}
$benchmarkContext = $null
if ($benchmarkRequested) {
    . (Join-Path $PSScriptRoot 'GX1-PowerBenchmarkScope.ps1')
}

if ($PolicySelfTest) {
    Invoke-Gx1PolicySelfTest
    return
}

try {
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "GX1 GPU guard config is unavailable: $configPath"
    }
    $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($config.schema_version -ne 'gx1_gpu_power_and_idle_guard_v2') {
        throw "GX1 GPU guard config schema mismatch: $($config.schema_version)"
    }
    if ([string]$config.expected_gpu_uuid -notmatch '^GPU-[0-9A-Fa-f-]+$' -or
        [string]$config.gpu_pnp_instance_id -notmatch '^PCI\\VEN_10DE&' -or
        [int]$config.power_limit_w -lt 100 -or [int]$config.power_limit_w -gt 160 -or
        [int]$config.sample_seconds -lt 1 -or
        [int]$config.idle_power_threshold_w -lt 40 -or [int]$config.idle_power_threshold_w -gt 120 -or
        [int]$config.idle_memory_max_mib -lt 128 -or [int]$config.idle_memory_max_mib -gt 2048 -or
        [int]$config.idle_utilization_max_percent -lt 0 -or [int]$config.idle_utilization_max_percent -gt 10 -or
        [int]$config.idle_required_samples -lt 12 -or
        [int]$config.max_recoveries_per_window -lt 1 -or [int]$config.max_recoveries_per_window -gt 4) {
        throw 'GX1 GPU guard config values are outside the fail-closed policy bounds'
    }
}
catch {
    $configurationFailure = $_.Exception.Message
    try {
        Stop-Gx1TelemetryBridge -Config ([pscustomobject]@{ telemetry_task_name = 'GX1HostTelemetryBridge' })
    }
    catch {
        Write-Gx1GuardLog "CONFIG_TELEMETRY_STOP_FAILURE message=$($_.Exception.Message)"
    }
    Write-Gx1GuardLog "CONFIG_FAILURE message=$configurationFailure telemetry_bridge=STOPPED"
    throw $configurationFailure
}

$highIdleSamples = 0
$lastPowerLimitCheck = [datetime]::MinValue
Write-Gx1GuardLog "GUARD_START gpu_uuid=$($config.expected_gpu_uuid) power_limit_w=$($config.power_limit_w) idle_power_threshold_w=$($config.idle_power_threshold_w) idle_required_samples=$($config.idle_required_samples) sample_seconds=$($config.sample_seconds)"
if (Test-Path -LiteralPath $blockerPath -PathType Leaf) {
    Stop-Gx1TelemetryBridge -Config $config
    Write-Gx1GuardLog 'GUARD_FAIL_CLOSED_ACTIVE telemetry_bridge=STOPPED'
}

foreach ($attempt in 1..([int]$config.initial_retry_count)) {
    try {
        $powerSample = Set-Gx1PowerLimit -Config $config
        $lastPowerLimitCheck = [datetime]::UtcNow
        Write-Gx1GuardLog "POWER_LIMIT_VERIFIED gpu_uuid=$($powerSample.gpu_uuid) power_limit_w=$($powerSample.power_limit_w) startup_attempt=$attempt"
        break
    }
    catch {
        Write-Gx1GuardLog "STARTUP_POWER_LIMIT_RETRY attempt=$attempt message=$($_.Exception.Message)"
        if ($attempt -lt [int]$config.initial_retry_count) {
            Start-Sleep -Seconds ([int]$config.retry_delay_seconds)
        }
    }
}

if ($benchmarkRequested) {
    if (Test-Path -LiteralPath $blockerPath) { throw 'Cannot arm a benchmark while the GPU guard is blocked' }
    $benchmarkContext = New-Gx1BenchmarkKeeperContext -ScopePath $BenchmarkScopePath `
        -ScopeSha256 $BenchmarkScopeSha256 -GuardRoot $root -BaselineConfig $config
}

try {
while ($true) {
    try {
        if (Test-Path -LiteralPath $blockerPath -PathType Leaf) {
            if ($null -ne $benchmarkContext) {
                Close-Gx1BenchmarkKeeperContext -Context $benchmarkContext -BaselineConfig $config -Reason guard_failure
            }
            Stop-Gx1TelemetryBridge -Config $config
        }
        $enforcementConfig = $config
        if ($null -ne $benchmarkContext) {
            Sync-Gx1BenchmarkKeeperContext -Context $benchmarkContext -BaselineConfig $config
            if ($benchmarkContext.Phase -eq 'active') { $enforcementConfig = $benchmarkContext.PowerConfig }
        }
        $now = [datetime]::UtcNow
        if (($now - $lastPowerLimitCheck).TotalSeconds -ge [int]$config.recheck_seconds) {
            $powerSample = Set-Gx1PowerLimit -Config $enforcementConfig
            $lastPowerLimitCheck = $now
            Write-Gx1GuardLog "POWER_LIMIT_VERIFIED gpu_uuid=$($powerSample.gpu_uuid) power_limit_w=$($powerSample.power_limit_w)"
        }
        $sample = Get-Gx1GpuSample -Config $config
        if ($null -ne $benchmarkContext -and $benchmarkContext.Phase -eq 'active') {
            Assert-Gx1ExactBenchmarkTreatment -Scope $benchmarkContext.Scope -Sample $sample
        }
        if (Test-Gx1HighIdleSample -Sample $sample -Config $config) {
            $highIdleSamples += 1
            if ($highIdleSamples -eq 1) {
                Write-Gx1GuardLog "HIGH_IDLE_SUSPECTED pstate=$($sample.pstate) power_draw_w=$($sample.power_draw_w) memory_used_mib=$($sample.memory_used_mib) utilization_percent=$($sample.utilization_percent)"
            }
        }
        else {
            if ($highIdleSamples -gt 0) {
                Write-Gx1GuardLog "HIGH_IDLE_CLEARED samples=$highIdleSamples pstate=$($sample.pstate) power_draw_w=$($sample.power_draw_w) memory_used_mib=$($sample.memory_used_mib) utilization_percent=$($sample.utilization_percent)"
            }
            $highIdleSamples = 0
        }

        if ($highIdleSamples -ge [int]$config.idle_required_samples) {
            if ($null -ne $benchmarkContext) {
                Close-Gx1BenchmarkKeeperContext -Context $benchmarkContext -BaselineConfig $config -Reason recovery
            }
            try {
                $now = [datetime]::UtcNow
                $windowStart = $now.AddSeconds(-[int]$config.recovery_window_seconds)
                $history = @(Get-Gx1RecoveryHistory | Where-Object { $_ -ge $windowStart })
                $lastRecovery = if ($history.Count -gt 0) { $history[-1] } else { [datetime]::MinValue }
                if ($history.Count -ge [int]$config.max_recoveries_per_window) {
                    Write-Gx1Blocker -Sample $sample -Reason 'GPU_IDLE_RECOVERY_RATE_LIMITED'
                    Stop-Gx1TelemetryBridge -Config $config
                    Write-Gx1GuardLog "RECOVERY_RATE_LIMITED attempts=$($history.Count) window_seconds=$($config.recovery_window_seconds) telemetry_bridge=STOPPED"
                }
                elseif (($now - $lastRecovery).TotalSeconds -lt [int]$config.recovery_cooldown_seconds) {
                    Write-Gx1Blocker -Sample $sample -Reason 'GPU_IDLE_RECOVERY_COOLDOWN'
                    Stop-Gx1TelemetryBridge -Config $config
                    Write-Gx1GuardLog "RECOVERY_COOLDOWN remaining_seconds=$([int]$config.recovery_cooldown_seconds - [int]($now - $lastRecovery).TotalSeconds) telemetry_bridge=STOPPED"
                }
                else {
                    $history += $now
                    Save-Gx1RecoveryHistory -History $history
                    Write-Gx1GuardLog "RECOVERY_START pstate=$($sample.pstate) power_draw_w=$($sample.power_draw_w) memory_used_mib=$($sample.memory_used_mib) utilization_percent=$($sample.utilization_percent)"
                    $afterSample = Invoke-Gx1GpuRecovery -Config $config -BeforeSample $sample
                    Write-Gx1GuardLog "RECOVERY_SUCCESS pstate=$($afterSample.pstate) power_draw_w=$($afterSample.power_draw_w) power_limit_w=$($afterSample.power_limit_w) memory_used_mib=$($afterSample.memory_used_mib) utilization_percent=$($afterSample.utilization_percent) telemetry_bridge=RESTARTED"
                }
            }
            catch {
                $failureMessage = $_.Exception.Message
                Write-Gx1Blocker -Sample $sample -Reason 'GPU_IDLE_RECOVERY_FAILED'
                try {
                    Stop-Gx1TelemetryBridge -Config $config
                }
                catch {
                    Write-Gx1GuardLog "TELEMETRY_STOP_FAILURE message=$($_.Exception.Message)"
                }
                Write-Gx1GuardLog "RECOVERY_FAILURE message=$failureMessage telemetry_bridge=STOPPED"
            }
            $highIdleSamples = 0
        }

        if ($Once) {
            [pscustomobject]@{
                schema_version = 'gx1_gpu_power_and_idle_guard_once_v1'
                decision = if (Test-Gx1HighIdleSample -Sample $sample -Config $config) { 'HIGH_IDLE_SAMPLE' } else { 'PASS' }
                sample = $sample
            } | ConvertTo-Json -Depth 4 -Compress
            return
        }
    }
    catch {
        $sampleFailure = $_.Exception.Message
        if ($null -ne $benchmarkContext -and $benchmarkContext.Phase -ne 'closed') {
            try {
                Close-Gx1BenchmarkKeeperContext -Context $benchmarkContext -BaselineConfig $config -Reason telemetry_failure
            }
            catch { Write-Gx1GuardLog "BENCHMARK_CLOSURE_FAILURE message=$($_.Exception.Message)" }
        }
        Write-Gx1GuardLog "SAMPLE_FAILURE message=$sampleFailure"
        if ($Once) {
            throw
        }
    }
    Start-Sleep -Seconds ([int]$config.sample_seconds)
}

}
finally {
    # Ordinary termination/uncaught loop failures must close before leaving.
    # OS-forced process death still requires the external restart/restore owner.
    if ($null -ne $benchmarkContext -and $benchmarkContext.Phase -ne 'closed') {
        Close-Gx1BenchmarkKeeperContext -Context $benchmarkContext -BaselineConfig $config -Reason guard_failure
    }
}
