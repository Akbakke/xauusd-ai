param([string]$ControlRoot = '/home/andre2/src/GX1_VAL_CONTROL_V22')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$source = (@(& wsl.exe -d Ubuntu-22.04 -u andre2 -- /bin/cat ($ControlRoot + '/scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1'))) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'control source read failed' }
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'controller syntax invalid' }
$SourceRepo = '/frozen/model'
$CampaignControlRepo = '/repaired/control'
$Python = '.venv/bin/python'
$invoke = $ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Invoke-Gx1Json'}, $true)
. ([ScriptBlock]::Create($invoke.Extent.Text))
function Invoke-Gx1WslBounded {
    param([string[]]$Arguments, [int]$TimeoutMilliseconds)
    if ($Arguments[1] -cne $CampaignControlRepo -or $TimeoutMilliseconds -ne 30000) {
        throw 'metadata execution repo or timeout changed incorrectly'
    }
    return [pscustomobject]@{ ExitCode = 0; StdOut = '{"ok":true}'; StdErr = '' }
}
$result = Invoke-Gx1Json -Arguments @('inspect')
if ($result.ok -ne $true) { throw 'metadata dispatch failed' }
$trainerStart = [regex]::Match($source, '(?m)^\$trainerArguments = [^\r\n]+')
$Distro = 'Ubuntu-22.04'
$LinuxUser = 'andre2'
$argv = @('runner')
. ([ScriptBlock]::Create($trainerStart.Value))
if ($trainerArguments[5] -cne $SourceRepo) { throw 'model execution left its frozen repo' }
$outcomeStart = $source.IndexOf('$outcome = if (')
$outcomeEnd = $source.IndexOf('$recorded = Invoke-Gx1Json', $outcomeStart)
$outcomeScript = [ScriptBlock]::Create($source.Substring($outcomeStart, $outcomeEnd - $outcomeStart))
foreach ($case in @(
    @{ trainer = 0; observer = 0; expected = 'RESUMABLE_OR_COMPLETE'; actual = 'AUTO' },
    @{ trainer = 0; observer = 0; expected = 'COMPLETE'; actual = 'COMPLETE' },
    @{ trainer = 7; observer = 0; expected = 'RESUMABLE_OR_COMPLETE'; actual = 'FAILED' },
    @{ trainer = 0; observer = 7; expected = 'RESUMABLE_OR_COMPLETE'; actual = 'FAILED' }
)) {
    $trainer = [pscustomobject]@{ ExitCode = $case.trainer }
    $observer = [pscustomobject]@{ ExitCode = $case.observer }
    $invocation = [pscustomobject]@{ expected_success_outcome = $case.expected }
    . $outcomeScript
    if ($outcome -cne $case.actual) { throw 'invalid resolved outcome request' }
}
$beginStart = $source.IndexOf('$begin = Invoke-Gx1Json')
$catchStart = $source.IndexOf('} catch {', $beginStart)
if ($beginStart -lt 0 -or $catchStart -lt $beginStart -or
    $source.IndexOf("Write-Gx1BootstrapErrorReceipt -Stage", $catchStart) -lt $catchStart) {
    throw 'begin failure is not covered by bootstrap receipt'
}
'VAL_CONTROL_PASS metadata_repo=fixed model_repo=frozen success_union=AUTO child_failures=FAILED begin_failure=recorded'

