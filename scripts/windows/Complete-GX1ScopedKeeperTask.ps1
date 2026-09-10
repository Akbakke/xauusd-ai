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
    $rows = @(& $NvidiaSmiPath --query-gpu=uuid,power.limit --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -ne 1) {
        throw 'Exactly one NVIDIA GPU sample is required'
    }
    $parts = @($rows[0].Split(',') | ForEach-Object { $_.Trim() })
    if ($parts.Count -ne 2) { throw 'Unexpected NVIDIA GPU sample shape' }
    return [pscustomobject]@{
        gpu_uuid = $parts[0]
        power_limit_w = [double]::Parse($parts[1], [Globalization.CultureInfo]::InvariantCulture)
    }
}

$scopeDirectory = Join-Path (Join-Path $GuardRoot 'Benchmarks') $ScopeId
$scopePath = Join-Path $scopeDirectory 'scope.json'
$closePath = Join-Path $scopeDirectory 'close.json'
$receiptPath = Join-Path $scopeDirectory 'receipt.json'
$baselineTaskName = 'GX1GpuPowerLimit'
$taskName = 'GX1ScopedKeeper-' + $ScopeId.Substring(0, 8)

foreach ($path in @($scopePath, $closePath, $receiptPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw 'Scope, close intent and closed receipt must all exist before task cleanup'
    }
}
if ((Get-FileHash -LiteralPath $scopePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ScopeSha256) {
    throw 'Scope bytes changed; refusing task cleanup'
}
$scope = Get-Content -LiteralPath $scopePath -Raw -Encoding UTF8 | ConvertFrom-Json
$close = Get-Content -LiteralPath $closePath -Raw -Encoding UTF8 | ConvertFrom-Json
$receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
$closeScopeId = $close.PSObject.Properties['scope_id']
if ($scope.scope_id -cne $ScopeId -or $scope.gpu_uuid -cne 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29' -or
    ($null -ne $closeScopeId -and $closeScopeId.Value -cne $ScopeId) -or
    $close.scope_sha256 -cne $ScopeSha256 -or
    $receipt.scope_id -cne $ScopeId -or $receipt.scope_sha256 -cne $ScopeSha256 -or
    $receipt.phase -cne 'closed' -or $receipt.requested_power_limit_w -ne 160 -or
    $receipt.baseline_restoration.decision -cne 'PASS_BASELINE_PHYSICALLY_RESTORED' -or
    $receipt.baseline_restoration.scope_id -cne $ScopeId -or
    $receipt.baseline_restoration.gpu_uuid -cne $scope.gpu_uuid -or
    [double]$receipt.baseline_restoration.observed_power_limit_w -ne 160.0) {
    throw 'Closed scope receipt does not prove the exact 160 W baseline restoration'
}

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
    $arguments = [string]$task.Actions[0].Arguments
    if ($arguments -notmatch [regex]::Escape($scopePath) -or
        $arguments -notmatch [regex]::Escape($ScopeSha256)) {
        throw 'Scheduled task is not bound to the exact closed scope'
    }
    if ($task.State -eq 'Running') {
        # The closed receipt above already proves that the keeper restored
        # 160 W. It is now a baseline-only infinite loop and can stop at once.
        Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
}

$otherRunning = @(Get-ScheduledTask -TaskName 'GX1ScopedKeeper-*' -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq 'Running' })
if ($otherRunning.Count -ne 0) {
    throw 'Another scoped keeper remains active; baseline task was not started'
}

$baselineTask = Get-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop
if ($baselineTask.State -ne 'Running') {
    Start-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop
}
$deadline = [datetime]::UtcNow.AddSeconds(15)
do {
    Start-Sleep -Milliseconds 250
    $baselineTask = Get-ScheduledTask -TaskName $baselineTaskName -ErrorAction Stop
} while ($baselineTask.State -ne 'Running' -and [datetime]::UtcNow -lt $deadline)
if ($baselineTask.State -ne 'Running') {
    throw 'Ordinary 160 W keeper did not enter Running state'
}
$gpu = Get-Gx1PhysicalPowerSample
if ($gpu.gpu_uuid -cne $scope.gpu_uuid -or $gpu.power_limit_w -ne 160.0) {
    throw 'Physical 160 W baseline is not proven after scoped task cleanup'
}

[pscustomobject]@{
    decision = 'PASS_SCOPED_KEEPER_COMPLETED_AT_BASELINE'
    scope_id = $ScopeId
    scope_sha256 = $ScopeSha256
    removed_task_name = $taskName
    baseline_task_name = $baselineTaskName
    baseline_task_state = [string]$baselineTask.State
    gpu = $gpu
    close_sha256 = (Get-FileHash -LiteralPath $closePath -Algorithm SHA256).Hash.ToLowerInvariant()
    receipt_sha256 = (Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json -Depth 6 -Compress
