param(
    [string]$ControllerSourceWsl = '/home/andre2/src/GX1_EXIT_LIFECYCLE_V2/scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1',
    [string]$DistroName = 'Ubuntu-22.04',
    [string]$LinuxUserName = 'andre2'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$lines = @(& wsl.exe -d $DistroName -u $LinuxUserName -- /bin/cat $ControllerSourceWsl)
if ($LASTEXITCODE -ne 0) { throw 'wsl cat failed' }
$source = $lines -join [Environment]::NewLine
[void][ScriptBlock]::Create($source)

# Bind against the real parameter block, without executing its host writes.
$parseTokens = $null
$parseErrors = $null
$controllerAst = [Management.Automation.Language.Parser]::ParseInput($source, [ref]$parseTokens, [ref]$parseErrors)
$bootFunction = $controllerAst.Find({
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Write-Gx1BootIdentity'
}, $true)
if ($null -eq $bootFunction) { throw 'boot identity function was not found' }
$bindingOnly = [ScriptBlock]::Create($bootFunction.Body.ParamBlock.Extent.Text + "`nreturn `$WslTimeoutMilliseconds")
if ((& $bindingOnly -WslTimeoutMilliseconds 30000) -ne 30000) {
    throw 'boot identity rejects the actual cold-start timeout'
}
foreach ($invalidTimeout in @(0, 30001)) {
    $rejectedTimeout = $false
    try { & $bindingOnly -WslTimeoutMilliseconds $invalidTimeout | Out-Null }
    catch [System.Management.Automation.ParameterBindingException] { $rejectedTimeout = $true }
    if (-not $rejectedTimeout) { throw "boot identity accepted invalid timeout: $invalidTimeout" }
}

# Exercise the real comparison predicate for zero and singleton differences.
$comparison = [regex]::Match($source, '(?m)^\s*if \((.+Compare-Object[^\r\n]+) -or\r?$')
if (-not $comparison.Success) { throw 'bridge configuration comparison was not found' }
$comparisonOnly = [ScriptBlock]::Create("param(`$expectedConfigurationFields, `$actualConfigurationFields)`n" + $comparison.Groups[1].Value)
$comparisonCases = @(
    @{ fields = @('beta', 'alpha'); rejected = $false },
    @{ fields = @('alpha'); rejected = $true },
    @{ fields = @('alpha', 'beta', 'unexpected'); rejected = $true }
)
foreach ($case in $comparisonCases) {
    $rejected = & $comparisonOnly -expectedConfigurationFields @('alpha', 'beta') -actualConfigurationFields $case.fields
    if ($rejected -isnot [bool] -or $rejected -ne $case.rejected) {
        throw 'bridge configuration field-set comparison failed'
    }
}

# Execute the production placement and writer against the selected local backend.
$statusRoot = [regex]::Match($source, '(?m)^\$runtimeWindows = [^\r\n]+')
if (-not $statusRoot.Success) { throw 'observer status root was not found' }
$statusRootOnly = [ScriptBlock]::Create("param(`$status)`n" + $statusRoot.Value + "`n`$runtimeWindows")
$actualStatusRoot = & $statusRootOnly -status @{ plan_sha256 = ('d' * 64) }
if (-not $actualStatusRoot.StartsWith('C:\') -or $actualStatusRoot.StartsWith('\\')) {
    throw 'observer status would use an unsupported filesystem'
}
$observerSourceWsl = $ControllerSourceWsl.Replace('GX1-RandomAccessCampaignV2Controller.ps1', 'GX1-RandomAccessCampaignV2Progress.ps1')
$observerSource = (@(& wsl.exe -d $DistroName -u $LinuxUserName -- /bin/cat $observerSourceWsl)) -join [Environment]::NewLine
if ($LASTEXITCODE -ne 0) { throw 'observer source read failed' }
$observerAst = [Management.Automation.Language.Parser]::ParseInput($observerSource, [ref]$parseTokens, [ref]$parseErrors)
$writer = $observerAst.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Write-Gx1AtomicJson'}, $true)
. ([ScriptBlock]::Create($writer.Extent.Text))
$testRoot = Join-Path $actualStatusRoot ('diagnostic-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
$statusPath = Join-Path $testRoot 'STATUS.json'
Write-Gx1AtomicJson -Path $statusPath -Value @{ sequence = 1 }
Write-Gx1AtomicJson -Path $statusPath -Value @{ sequence = 2 }
if ((Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json).sequence -ne 2) { throw 'second atomic status write failed' }

# Start-Process without retaining Handle returns null ExitCode on the actual host.
$trainer = Start-Process powershell.exe -ArgumentList @('-NoProfile', '-NonInteractive', '-Command', '"Start-Sleep -Milliseconds 500; exit 0"') -PassThru -NoNewWindow
$trainerRetention = [regex]::Match($source, '(?m)^\s*\$trainerHandle = [^\r\n]+')
. ([ScriptBlock]::Create($trainerRetention.Value))
$observer = Start-Process powershell.exe -ArgumentList @('-NoProfile', '-NonInteractive', '-Command', '"Start-Sleep -Milliseconds 500; exit 7"') -PassThru -NoNewWindow
$observerRetention = [regex]::Match($source, '(?m)^\$observerHandle = [^\r\n]+')
. ([ScriptBlock]::Create($observerRetention.Value))
$trainer.WaitForExit()
$observer.WaitForExit()
if ($null -eq $trainer.ExitCode -or $trainer.ExitCode -ne 0 -or $null -eq $observer.ExitCode -or $observer.ExitCode -ne 7) {
    throw "child exit codes were not preserved exactly: trainer=$($trainer.ExitCode) observer=$($observer.ExitCode)"
}

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
    "timeout_ms=$($timeoutClock.ElapsedMilliseconds) one_shot_initial_state=PASS boot_identity_parameter_binding=PASS bridge_configuration_comparison=PASS observer_checkpoint_isolation=PASS observer_local_atomic_replace=PASS retained_child_exit_codes=PASS"
)
