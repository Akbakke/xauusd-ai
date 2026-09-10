param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')]
    [string]$ScopeId,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ScopeSha256,
    [string]$GuardRoot = 'C:\ProgramData\GX1\GpuPowerLimit',
    [string]$NvidiaSmiPath = 'C:\Windows\System32\nvidia-smi.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Gx1PhysicalPowerSample {
    $rows = @(& $NvidiaSmiPath --query-gpu=uuid,power.limit,power.draw,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -ne 1) {
        throw 'Exactly one NVIDIA GPU sample is required'
    }
    $parts = @($rows[0].Split(',') | ForEach-Object { $_.Trim() })
    if ($parts.Count -ne 6) { throw 'Unexpected NVIDIA GPU sample shape' }
    return [pscustomobject]@{
        gpu_uuid = $parts[0]
        power_limit_w = [double]::Parse($parts[1], [Globalization.CultureInfo]::InvariantCulture)
        power_draw_w = [double]::Parse($parts[2], [Globalization.CultureInfo]::InvariantCulture)
        core_temperature_c = [int]$parts[3]
        memory_used_mib = [int]$parts[4]
        utilization_percent = [int]$parts[5]
    }
}

$scopeDirectory = Join-Path (Join-Path $GuardRoot 'Benchmarks') $ScopeId
$scopePath = Join-Path $scopeDirectory 'scope.json'
$receiptPath = Join-Path $scopeDirectory 'receipt.json'
$closePath = Join-Path $scopeDirectory 'close.json'
$keeperPath = Join-Path $GuardRoot 'GX1-ApplyGpuPowerLimit.ps1'
$baselineTaskName = 'GX1GpuPowerLimit'
$taskName = 'GX1ScopedKeeper-' + $ScopeId.Substring(0, 8)

if (-not (Test-Path -LiteralPath $scopePath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $keeperPath -PathType Leaf)) {
    throw 'Scope and installed keeper must be regular files'
}
if ((Get-FileHash -LiteralPath $scopePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ScopeSha256) {
    throw 'Scoped keeper source changed'
}
$scope = Get-Content -LiteralPath $scopePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($scope.scope_id -cne $ScopeId -or
    $scope.gpu_uuid -cne 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29' -or
    $scope.target_power_limit_w -notin @(160, 200) -or
    $scope.baseline_power_limit_w -ne 160) {
    throw 'Scoped keeper identity, GPU or power treatment is invalid'
}
if (Test-Path -LiteralPath $closePath) {
    throw 'A closed scope cannot be started'
}

$otherScopedTasks = @(Get-ScheduledTask -TaskName 'GX1ScopedKeeper-*' -ErrorAction SilentlyContinue |
    Where-Object { $_.TaskName -cne $taskName })
if ($otherScopedTasks.Count -ne 0) {
    throw 'Another scoped keeper task must be completed before a new scope starts'
}
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'The exact scoped keeper task already exists; inspect or complete it instead of replacing it'
}

$registered = $false
try {
    $baselineTask = Get-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop
    if ($baselineTask.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop
    }

    $deadline = [datetime]::UtcNow.AddSeconds(20)
    do {
        $old = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
            Where-Object {
                $_.ProcessId -ne $PID -and $null -ne $_.CommandLine -and
                $_.CommandLine -match '-File.*GX1-ApplyGpuPowerLimit.ps1'
            })
        if ($old.Count) { Start-Sleep -Milliseconds 250 }
    } while ($old.Count -and [datetime]::UtcNow -lt $deadline)
    if ($old.Count) { throw 'Baseline or stale scoped keeper did not stop' }

    $argumentText = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
        $keeperPath + '" -BenchmarkScopePath "' + $scopePath +
        '" -BenchmarkScopeSha256 "' + $ScopeSha256 + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argumentText
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings | Out-Null
    $registered = $true
    Start-ScheduledTask -TaskName $taskName

    $deadline = [datetime]::UtcNow.AddSeconds(30)
    $receipt = $null
    do {
        Start-Sleep -Milliseconds 250
        if (Test-Path -LiteralPath $receiptPath) {
            try { $receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json }
            catch { $receipt = $null }
        }
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    } while (($null -eq $receipt -or $receipt.phase -cne 'active') -and
             $task.State -eq 'Running' -and [datetime]::UtcNow -lt $deadline)

    if ($null -eq $receipt -or $receipt.phase -cne 'active' -or
        $receipt.scope_id -cne $ScopeId -or $receipt.scope_sha256 -cne $ScopeSha256 -or
        $receipt.gpu_uuid -cne $scope.gpu_uuid -or
        $receipt.requested_power_limit_w -ne $scope.target_power_limit_w -or
        ([datetime]::UtcNow - [datetime]::Parse($receipt.observed_utc).ToUniversalTime()).TotalSeconds -gt 5) {
        throw 'Fresh active scoped receipt was not established'
    }
    $keeper = Get-CimInstance Win32_Process -Filter "ProcessId=$($receipt.keeper_pid)"
    if ($null -eq $keeper -or $null -eq $keeper.CommandLine -or
        $keeper.CommandLine -notmatch [regex]::Escape($scopePath) -or
        $keeper.CommandLine -notmatch [regex]::Escape($ScopeSha256)) {
        throw 'Receipt keeper process is absent or not bound to the exact scope'
    }
    $gpu = Get-Gx1PhysicalPowerSample
    if ($gpu.gpu_uuid -cne $scope.gpu_uuid -or $gpu.power_limit_w -ne [double]$scope.target_power_limit_w) {
        throw 'Physical GPU treatment does not match the active scope'
    }

    [pscustomobject]@{
        decision = 'PASS_PERSISTENT_SCOPED_KEEPER_ACTIVE'
        task_name = $taskName
        task_state = [string](Get-ScheduledTask -TaskName $taskName -ErrorAction Stop).State
        keeper_pid = $receipt.keeper_pid
        scope_id = $ScopeId
        scope_sha256 = $ScopeSha256
        receipt = $receipt
        gpu = $gpu
        baseline_task_state = [string](Get-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop).State
    } | ConvertTo-Json -Depth 8 -Compress
}
catch {
    $failure = $_.Exception.Message
    if ($registered) {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    }
    $restoration = $null
    try {
        $restoration = (& $keeperPath -Once) | ConvertFrom-Json
        if ($restoration.decision -cne 'PASS' -or
            $restoration.sample.gpu_uuid -cne $scope.gpu_uuid -or
            [double]$restoration.sample.power_limit_w -ne 160.0) {
            throw 'Emergency baseline restoration was not proven'
        }
        Start-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop
    }
    catch {
        throw "$failure; scoped task stopped, but 160 W baseline restoration failed: $($_.Exception.Message)"
    }
    throw "$failure; scoped task stopped and physical 160 W baseline restored; close the consumed scope before reuse"
}
