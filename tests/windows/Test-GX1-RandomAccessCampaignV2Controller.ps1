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
$initialCall = $source.LastIndexOf('$initial = Get-Gx1InitialCampaignState')
$bootUse = $source.IndexOf('$boot = $initial.Boot', $initialCall)
$statusUse = $source.IndexOf('$status = $initial.Status', $bootUse)
if ($initialCall -lt 0 -or $bootUse -le $initialCall -or $statusUse -le $bootUse) {
    throw 'top-level cold-WSL gate must precede boot and campaign state use'
}
$start = $source.IndexOf('function Get-Gx1StringSha256')
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
function Start-Sleep { [CmdletBinding()] param([int]$Milliseconds, [int]$Seconds) }
$script:initialWriteCalls = 0
$script:initialInspectCalls = 0
$script:recoveryCalls = 0
$script:lastRecoverySignature = $null
function Write-Gx1BootIdentity {
    param([int]$WslTimeoutMilliseconds)
    $script:initialWriteCalls++
    if ($script:initialWriteCalls -lt 3) { throw 'Wsl/Service/E_UNEXPECTED' }
    return [pscustomobject]@{ Linux = '/mnt/c/ProgramData/GX1/RandomAccessCampaignV2/CURRENT_BOOT.json' }
}
function Invoke-Gx1Json {
    param([string[]]$Arguments, [int]$TimeoutMilliseconds)
    $script:initialInspectCalls++
    return [pscustomobject]@{ ok = $true }
}
function Invoke-Gx1WslBootstrapRecovery {
    param([string]$FailureSignature)
    $script:recoveryCalls++
    $script:lastRecoverySignature = $FailureSignature
}
$PlanJson = '/tmp/plan.json'
$PlanFileSha256 = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
$Distro = $DistroName
$ExpectedControllerSha256 = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
$initial = Get-Gx1InitialCampaignState
if ($script:initialWriteCalls -ne 3 -or $script:initialInspectCalls -ne 1 -or $script:recoveryCalls -ne 1 -or $initial.Status.ok -ne $true -or $script:lastRecoverySignature -cne 'Wsl/Service/E_UNEXPECTED') {
    throw "cold WSL retry write_calls=$script:initialWriteCalls inspect_calls=$script:initialInspectCalls recovery_calls=$script:recoveryCalls"
}

$script:initialWriteCalls = 0
$script:initialInspectCalls = 0
$script:recoveryCalls = 0
$script:lastRecoverySignature = $null
function Write-Gx1BootIdentity {
    param([int]$WslTimeoutMilliseconds)
    $script:initialWriteCalls++
    if ($script:initialWriteCalls -lt 3) {
        throw "Bounded process timed out after 8000ms: $env:WINDIR\System32\wsl.exe"
    }
    return [pscustomobject]@{ Linux = '/mnt/c/ProgramData/GX1/RandomAccessCampaignV2/CURRENT_BOOT.json' }
}
$initial = Get-Gx1InitialCampaignState
if ($script:initialWriteCalls -ne 3 -or $script:initialInspectCalls -ne 1 -or $script:recoveryCalls -ne 1 -or $initial.Status.ok -ne $true -or $script:lastRecoverySignature -cne 'BOUNDED_WSL_EXE_TIMEOUT') {
    throw 'exact bounded wsl.exe timeout did not enter one recovery'
}

$script:recoveryCalls = 0
function Write-Gx1BootIdentity {
    param([int]$WslTimeoutMilliseconds)
    throw "Bounded process timed out after 8000ms: $env:WINDIR\System32\ping.exe"
}
$nonExactRejected = $false
try { Get-Gx1InitialCampaignState | Out-Null }
catch {
    if ($_.Exception.Message -like 'Initial campaign state failed without authorized WSL recovery signature*') {
        $nonExactRejected = $true
    } else { throw }
}
if (-not $nonExactRejected -or $script:recoveryCalls -ne 0) {
    throw 'non-wsl timeout entered recovery'
}

$bootstrapErrorRoot = Join-Path $env:TEMP ('gbe-' + [IO.Path]::GetRandomFileName())
try {
    $bootstrapException = [InvalidOperationException]::new('synthetic bootstrap failure')
    $bootstrapReceiptPath = Write-Gx1BootstrapErrorReceipt `
        -Stage 'initial_campaign_state' -Exception $bootstrapException `
        -ControllerPath $PSCommandPath -ErrorRoot $bootstrapErrorRoot
    $bootstrapReceiptSha256 = (Get-FileHash -LiteralPath $bootstrapReceiptPath -Algorithm SHA256).Hash
    $bootstrapReceipt = Get-Content -LiteralPath $bootstrapReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($bootstrapReceipt.schema_version -cne 'gx1_campaign_bootstrap_error_receipt_v1' -or
        $bootstrapReceipt.stage -cne 'initial_campaign_state' -or
        $bootstrapReceipt.exception_type -cne 'System.InvalidOperationException' -or
        $bootstrapReceipt.exception_message -cne 'synthetic bootstrap failure') {
        throw 'bootstrap error receipt lost exact error evidence'
    }
    $overwriteRejected = $false
    try {
        Write-Gx1BootstrapErrorReceipt `
            -Stage 'telemetry_readiness' -Exception ([Exception]::new('replacement')) `
            -ControllerPath $PSCommandPath -ErrorRoot $bootstrapErrorRoot | Out-Null
    } catch {
        if ($_.Exception.Message -like 'Bootstrap error receipt already exists*') {
            $overwriteRejected = $true
        } else { throw }
    }
    if (-not $overwriteRejected -or
        (Get-FileHash -LiteralPath $bootstrapReceiptPath -Algorithm SHA256).Hash -cne $bootstrapReceiptSha256) {
        throw 'bootstrap error receipt overwrite protection failed'
    }
} finally {
    Remove-Item -LiteralPath $bootstrapErrorRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$intentRoot = Join-Path $env:TEMP ('gx1-wsl-recovery-' + [guid]::NewGuid().ToString('N'))
try {
    $intent = New-Gx1WslRecoveryIntent -BootId 357 -FailureSignature 'Wsl/Service/E_UNEXPECTED' -RecoveryRoot $intentRoot
    if (-not (Test-Path -LiteralPath $intent.Path -PathType Leaf)) { throw 'recovery intent missing' }
    $duplicateRejected = $false
    try { New-Gx1WslRecoveryIntent -BootId 357 -FailureSignature 'Wsl/Service/E_UNEXPECTED' -RecoveryRoot $intentRoot | Out-Null }
    catch {
        if ($_.Exception.Message -like 'WSL recovery was already attempted*') { $duplicateRejected = $true } else { throw }
    }
    if (-not $duplicateRejected) { throw 'duplicate recovery intent accepted' }
} finally {
    Remove-Item -LiteralPath $intentRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$priorProgramData = $env:ProgramData
$recoveryProgramData = Join-Path $env:TEMP ('gwr-' + [IO.Path]::GetRandomFileName())
try {
    $env:ProgramData = $recoveryProgramData
    # Earlier scenarios replace recovery functions with counting mocks.
    # Reload the production recovery block before applying this scenario's targeted mocks.
    $recoveryStart = $source.IndexOf('function New-Gx1WslRecoveryIntent')
    $recoveryEnd = $source.IndexOf('function Get-Gx1InitialCampaignState', $recoveryStart)
    if ($recoveryStart -lt 0 -or $recoveryEnd -le $recoveryStart) { throw 'recovery function boundaries not found' }
    . ([ScriptBlock]::Create($source.Substring($recoveryStart, $recoveryEnd - $recoveryStart)))
    $bootRoot = Join-Path $recoveryProgramData 'GX1\RandomAccessCampaignV2'
    New-Item -ItemType Directory -Path $bootRoot -Force | Out-Null
    [IO.File]::WriteAllText(
        (Join-Path $bootRoot 'CURRENT_BOOT.json'),
        '{"schema_version":"gx1_windows_boot_identity_v1","boot_id":357}',
        [Text.UTF8Encoding]::new($false)
    )
    $script:controlCalls = 0
    $script:probeCalls = 0
    function Assert-Gx1ExclusiveWslRecoveryScope {}
    function Invoke-Gx1WslControlBounded {
        param([string[]]$Arguments, [int]$TimeoutMilliseconds = 10000)
        $script:controlCalls++
        if ($Arguments[0] -ceq '--terminate') {
            throw "Bounded process timed out after 10000ms: $env:WINDIR\System32\wsl.exe"
        }
        return [pscustomobject]@{ ExitCode = 0; StdOut = ''; StdErr = '' }
    }
    function Invoke-Gx1WslRecoveryProbe {
        $script:probeCalls++
        if ($script:probeCalls -eq 1) {
            throw "Bounded process timed out after 10000ms: $env:WINDIR\System32\wsl.exe"
        }
        return [pscustomobject]@{ ExitCode = 0; StdOut = ''; StdErr = '' }
    }
    Invoke-Gx1WslBootstrapRecovery -FailureSignature 'BOUNDED_WSL_EXE_TIMEOUT'
    $resultPath = Get-ChildItem -LiteralPath (Join-Path $bootRoot 'wsl-recovery') -Filter '*.result.json' -File
    $recoveryResult = Get-Content -LiteralPath $resultPath.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($script:controlCalls -ne 2 -or $script:probeCalls -ne 2 -or
        $recoveryResult.failure_signature -cne 'BOUNDED_WSL_EXE_TIMEOUT' -or
        $recoveryResult.terminate_timed_out -ne $true -or
        $recoveryResult.terminate_probe_timed_out -ne $true -or
        $recoveryResult.shutdown_attempted -ne $true -or
        $recoveryResult.outcome -cne 'PASS_RECOVERED') {
        throw 'bounded WSL timeout did not produce the single shutdown fallback'
    }
} finally {
    $env:ProgramData = $priorProgramData
    Remove-Item -LiteralPath $recoveryProgramData -Recurse -Force -ErrorAction SilentlyContinue
}

$script:taskCalls = 0
$script:addressCalls = 0
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
        $script:addressCalls++
        if ($script:addressCalls -lt 3) {
            return [pscustomobject]@{ ExitCode = -1; StdOut = ''; StdErr = 'Wsl/Service/E_UNEXPECTED' }
        }
        return [pscustomobject]@{ ExitCode = 0; StdOut = "172.30.231.75 `n"; StdErr = '' }
    }
    return [pscustomobject]@{ ExitCode = 0; StdOut = "default via 172.30.224.1 dev eth0 proto kernel `n"; StdErr = '' }
}
$Distro = $DistroName
$LinuxUser = $LinuxUserName
Wait-Gx1HostTelemetryBridgeV4BootReady
if ($script:taskCalls -ne 5 -or $script:addressCalls -ne 3) {
    throw "transient readiness task_calls=$script:taskCalls address_calls=$script:addressCalls"
}

Write-Output (
    'POWERSHELL_HARDENING_PASS ' +
    "large_stdout=$($large.StdOut.Length) large_stderr=$($large.StdErr.Length) " +
    "timeout_ms=$($timeoutClock.ElapsedMilliseconds) cold_wsl_attempts=$script:initialWriteCalls transient_task_attempts=$script:taskCalls transient_wsl_attempts=$script:addressCalls"
)
