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
$start = $source.IndexOf('function Join-Gx1NativeArguments')
$end = $source.IndexOf('function Reset-Gx1HostTelemetryPortProxy', $start)
if ($start -lt 0 -or $end -le $start) { throw 'function boundaries not found' }
. ([ScriptBlock]::Create($source.Substring($start, $end - $start)))

$rejected = $false
try { Join-Gx1NativeArguments -Arguments @('unsafe argument') | Out-Null }
catch { if ($_.Exception.Message -like 'Native process argument*') { $rejected = $true } else { throw } }
if (-not $rejected) { throw 'unsafe argument was accepted' }

$emitter = 'C:\Users\Andre\gx1_emit_large.ps1'
[IO.File]::WriteAllText(
    $emitter,
    "[Console]::Out.Write('x' * 200000); [Console]::Error.Write('y' * 200000)",
    [Text.UTF8Encoding]::new($false)
)
try {
    $large = Invoke-Gx1NativeProcessBounded `
        -FilePath (Join-Path $PSHome 'powershell.exe') `
        -ArgumentList @('-NoProfile', '-NonInteractive', '-File', $emitter) `
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

$waitStart = $source.IndexOf('function Wait-Gx1HostTelemetryBridgeV4BootReady')
. ([ScriptBlock]::Create($source.Substring($waitStart, $end - $waitStart)))
$script:taskCalls = 0
function Get-ScheduledTask {
    [CmdletBinding()] param([string]$TaskName)
    $script:taskCalls++
    if ($script:taskCalls -lt 3) { return [pscustomobject]@{ State = 'Ready' } }
    return [pscustomobject]@{ State = 'Running' }
}
function Get-NetTCPConnection {
    [CmdletBinding()] param([string]$State, [int]$LocalPort)
    return [pscustomobject]@{ LocalAddress = '127.0.0.1' }
}
function Invoke-Gx1WslBounded {
    param([string[]]$Arguments, [int]$TimeoutMilliseconds)
    if ($Arguments -contains '/bin/hostname') {
        return [pscustomobject]@{ ExitCode = 0; StdOut = "172.30.231.75 `n"; StdErr = '' }
    }
    return [pscustomobject]@{ ExitCode = 0; StdOut = "default via 172.30.224.1 dev eth0 proto kernel `n"; StdErr = '' }
}
function Start-Sleep { [CmdletBinding()] param([int]$Milliseconds) }
$Distro = $DistroName
$LinuxUser = $LinuxUserName
Wait-Gx1HostTelemetryBridgeV4BootReady
if ($script:taskCalls -ne 3) { throw "transient readiness calls=$script:taskCalls" }

Write-Output (
    'POWERSHELL_HARDENING_PASS ' +
    "large_stdout=$($large.StdOut.Length) large_stderr=$($large.StdErr.Length) " +
    "timeout_ms=$($timeoutClock.ElapsedMilliseconds) transient_attempts=$script:taskCalls"
)
