param(
    [string]$ControllerSourceWsl = '/home/andre2/src/GX1_EXIT_LIFECYCLE_V2/scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1',
    [string]$DistroName = 'Ubuntu-22.04',
    [string]$LinuxUserName = 'andre2'
)
$ErrorActionPreference = 'Stop'
$lines = @(& wsl.exe -d $DistroName -u $LinuxUserName -- /bin/cat $ControllerSourceWsl)
if ($LASTEXITCODE -ne 0) { throw 'wsl cat failed' }
$source = $lines -join [Environment]::NewLine
[void][ScriptBlock]::Create($source)

$helperStart = $source.IndexOf('function Join-Gx1NativeArguments')
$helperEnd = $source.IndexOf('function Reset-Gx1HostTelemetryPortProxy', $helperStart)
if ($helperStart -lt 0 -or $helperEnd -le $helperStart) { throw 'function boundaries not found' }
. ([ScriptBlock]::Create($source.Substring($helperStart, $helperEnd - $helperStart)))

$rejected = $false
try { Join-Gx1NativeArguments -Arguments @('unsafe argument') | Out-Null }
catch { if ($_.Exception.Message -like 'Native process argument*') { $rejected = $true } else { throw } }
if (-not $rejected) { throw 'unsafe argument was accepted' }

$emitter = Join-Path $env:TEMP ('gx1_emit_large_' + [guid]::NewGuid().ToString('N') + '.ps1')
[IO.File]::WriteAllText(
    $emitter,
    "[Console]::Out.Write(('x' * 200000 -join '')); [Console]::Error.Write(('y' * 200000 -join ''))",
    [Text.UTF8Encoding]::new($false)
)
try {
    $large = Invoke-Gx1NativeProcessBounded `
        -FilePath (Join-Path $PSHome 'powershell.exe') `
        -ArgumentList @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $emitter) `
        -TimeoutMilliseconds 8000
    if ($large.ExitCode -ne 0 -or $large.StdOut.Length -ne 200000 -or $large.StdErr.Length -ne 200000) {
        throw "pipe drain failed: exit=$($large.ExitCode) out=$($large.StdOut.Length) err=$($large.StdErr.Length)"
    }
} finally {
    Remove-Item -LiteralPath $emitter -Force -ErrorAction SilentlyContinue
}

$timeoutClock = [Diagnostics.Stopwatch]::StartNew()
$timedOut = $false
try {
    Invoke-Gx1NativeProcessBounded `
        -FilePath (Join-Path $env:WINDIR 'System32\ping.exe') `
        -ArgumentList @('127.0.0.1', '-n', '6', '-w', '1000') `
        -TimeoutMilliseconds 100 | Out-Null
} catch {
    if ($_.Exception.Message -like 'Bounded process timed out*') { $timedOut = $true } else { throw }
}
if (-not $timedOut -or $timeoutClock.ElapsedMilliseconds -gt 3000) {
    throw "bounded timeout failed: elapsed=$($timeoutClock.ElapsedMilliseconds)"
}

$initialStart = $source.IndexOf('function Get-Gx1InitialCampaignState')
$initialEnd = $source.IndexOf('function Wait-Gx1HostTelemetryBridgeV4BootReady', $initialStart)
if ($initialStart -lt 0 -or $initialEnd -le $initialStart) { throw 'initial-state function boundaries not found' }
. ([ScriptBlock]::Create($source.Substring($initialStart, $initialEnd - $initialStart)))
$script:bootCalls = 0
$script:inspectCalls = 0
function Write-Gx1BootIdentity {
    param([int]$WslTimeoutMilliseconds)
    $script:bootCalls++
    if ($WslTimeoutMilliseconds -ne 30000) { throw "wrong first WSL timeout: $WslTimeoutMilliseconds" }
    return [pscustomobject]@{ Linux = '/tmp/CURRENT_BOOT.json' }
}
function Invoke-Gx1Json {
    param([string[]]$Arguments, [int]$TimeoutMilliseconds)
    $script:inspectCalls++
    if ($TimeoutMilliseconds -ne 30000) { throw "wrong inspect timeout: $TimeoutMilliseconds" }
    return [pscustomobject]@{ ok = $true; result = [pscustomobject]@{} }
}
$PlanJson = '/tmp/plan.json'
$PlanFileSha256 = ('0' * 64)
$Distro = $DistroName
$LinuxUser = $LinuxUserName
$initial = Get-Gx1InitialCampaignState
if ($script:bootCalls -ne 1 -or $script:inspectCalls -ne 1 -or $initial.Status.ok -ne $true) {
    throw "one_shot_initial_state boot_calls=$script:bootCalls inspect_calls=$script:inspectCalls"
}

Write-Output (
    'POWERSHELL_HARDENING_PASS ' +
    "large_stdout=$($large.StdOut.Length) large_stderr=$($large.StdErr.Length) " +
    "timeout_ms=$($timeoutClock.ElapsedMilliseconds) one_shot_initial_state=PASS"
)
