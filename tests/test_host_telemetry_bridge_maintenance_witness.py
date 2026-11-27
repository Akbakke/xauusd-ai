"""Actual PowerShell owner functions, synthetic CIM/registry/IO; never installer/service launch."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / 'scripts/windows/Install-GX1-HostTelemetryBridge.ps1'
POWERSHELL = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')


def service_source():
    return INSTALLER.read_text().split("$serviceSource = @'\n", 1)[1].split("\n'@\n", 1)[0]


def test_logging_preserves_every_existing_installer_statement_and_wire_byte():
    source = INSTALLER.read_text()
    before = subprocess.check_output(['git', 'show', '5a5cbd37e:scripts/windows/Install-GX1-HostTelemetryBridge.ps1'], cwd=REPO, text=True)
    start = source.index('function New-MaintenanceWitnessSnapshot {\n')
    stop = source.index('function Get-HostResourceTelemetry {\n', start)
    normalized = source[:start] + source[stop:]
    additions = [
        '$script:MaintenanceWitnessState = $null\n',
        '        maintenance_snapshot = New-MaintenanceWitnessSnapshot -Workers $workers -Setup $setup -CbsPending $cbs -WuRequired $wu -Rebooting $rebooting\n',
        "            $null = Write-MaintenanceWitness -HostResources $hostResources -Nonce $nonce -ObservedMonotonicMs $observed -JournalPath (Join-Path $PSScriptRoot 'GX1-HostTelemetryBridgeMaintenance.jsonl')\n",
    ]
    for addition in additions:
        assert normalized.count(addition) == 1
        normalized = normalized.replace(addition, '', 1)
    assert normalized == before
    service = service_source()
    parts = {
        'payload': service.split('$payload = @(\n', 1)[1].split(') -join "`n"', 1)[0],
        'wire': next(line for line in service.splitlines() if line.strip().startswith('$json = ')),
        'predicate': next(line for line in service.split('function Get-HostResourceTelemetry {', 1)[1].splitlines() if line.strip().startswith('maintenance_active = ')),
        'workers': next(line for line in service.splitlines() if line.strip().startswith('$workers = ')),
    }
    assert {name: hashlib.sha256(value.encode()).hexdigest() for name, value in parts.items()} == {
        'payload': '657ace5e0fbe84d4e415ada9a97d7055f064a6bcf36c8b65d53014858936c7ec',
        'wire': 'e349312f5ba66b539cf54ab6b86137dab1eb491256f1f99fdd1a7af4bd8289d3',
        'predicate': '46cb473a145d00feb2f8ce98acce6fbfa74789ae5da8b8b0f82e3b69fab29172',
        'workers': '832bf36adce6b6af5bc0df7c213157b9ff2357a3da9e2d98f915a1aeb9a78b1e',
    }


def test_witness_is_local_before_200_and_has_no_retention_or_source_requery():
    service = service_source()
    writer = service.split('function Write-MaintenanceWitness {', 1)[1].split('function Get-HostResourceTelemetry {', 1)[0]
    assert 'Get-CimInstance' not in writer and 'Get-ItemProperty' not in writer and 'Test-Path' not in writer
    assert 'OpenOrCreate' in writer and 'FileMode]::Append' not in writer
    assert '.Flush($true)' in writer and 'Maintenance witness readback differs' in writer
    assert writer.index('.Flush($true)') < writer.index('$script:MaintenanceWitnessState = $stateJson')
    assert 'SetLength' not in writer and 'Delete(' not in writer and 'Remove-Item' not in writer
    assert '8MiB' not in writer and 'MaxLength' not in writer
    request = service.split('$nonce = Require-RequestNonce -Request $context.Request', 1)[1]
    assert request.index('$hostResources = Get-HostResourceTelemetry') < request.index('$null = Write-MaintenanceWitness ')
    assert request.index('$null = Write-MaintenanceWitness ') < request.index('-StatusCode 200')
    assert '-StatusCode 503' in request


HARNESS = r"""
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$WarningPreference='SilentlyContinue'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$source=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__SERVICE_BASE64__'))
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseInput($source,[ref]$tokens,[ref]$errors)
if($errors.Count -ne 0){throw ('Service AST invalid: '+($errors|Out-String))}
$wanted=@('New-MaintenanceWitnessSnapshot','ConvertTo-MaintenanceWitnessJson','Write-MaintenanceWitness','Get-HostResourceTelemetry','Format-CanonicalFloat')
$functions=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in $wanted},$true))
if($functions.Count -ne $wanted.Count){throw 'Exact function inventory missing'}
foreach($function in $functions){Invoke-Expression $function.Extent.Text}
$handlers=@($ast.FindAll({param($node) $node -is [Management.Automation.Language.TryStatementAst] -and $node.Body.Statements[0].Extent.Text -eq '$nonce = Require-RequestNonce -Request $context.Request'},$true))
if($handlers.Count -ne 1){throw 'Exact request handler missing'}
$script:Handler=$handlers[0].Extent.Text.Replace('$PSScriptRoot','$script:TestJournalRoot')
$script:HostPhysicalTotalBytes=[uint64](64GB)
$invariant=[Globalization.CultureInfo]::InvariantCulture
$responseSchema='gx1_host_resource_telemetry_v2'
$script:Passed=[Collections.Generic.List[string]]::new()
$script:Base=[IO.Path]::Combine([IO.Path]::GetTempPath(),'GX1_MaintenanceSynthetic_'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($script:Base)
function Assert-Test($ok,$reason){if(-not $ok){throw ('Synthetic maintenance failure: '+$reason)}}
function New-Case {
    return [pscustomobject]@{Workers=@();Setup=[pscustomobject]@{SystemSetupInProgress=0;OOBEInProgress=0};Cbs=$false;Wu=$false;Rebooting=$false}
}
function New-Worker($name='TiWorker.exe') {
    return [pscustomobject]@{Name=$name;ProcessId=[uint32]123;ParentProcessId=[uint32]4;CreationDate=[datetime]::Parse('2026-11-27T17:44:40Z').ToUniversalTime()}
}
function Get-CimInstance {
    param($ClassName,$Filter,$ErrorAction)
    if($ClassName -eq 'Win32_PerfRawData_PerfOS_Memory'){$script:Counts.memory++;return [pscustomobject]@{AvailableBytes=[uint64](40GB);CommittedBytes=[uint64](20GB);CommitLimit=[uint64](68GB)}}
    if($ClassName -eq 'Win32_Process'){
        $script:Counts.workers++
        Assert-Test ($Filter -eq "Name='TiWorker.exe' OR Name='MoUsoCoreWorker.exe' OR Name='SetupHost.exe' OR Name='SetupPrep.exe' OR Name='WindowsUpdateBox.exe' OR Name='wuauclt.exe'") 'worker filter changed'
        return $script:Case.Workers
    }
    throw 'Unexpected live CIM query'
}
function Get-ItemProperty {
    param($LiteralPath,$ErrorAction)
    Assert-Test ($LiteralPath -eq 'HKLM:\SYSTEM\Setup') 'unexpected registry source'
    $script:Counts.setup++;return $script:Case.Setup
}
function Test-Path {
    param($LiteralPath,$ErrorAction)
    $script:Counts.registry++
    switch -Wildcard($LiteralPath){
        '*\RebootPending'{return $script:Case.Cbs}
        '*\RebootRequired'{return $script:Case.Wu}
        '*\RebootInProgress'{return $script:Case.Rebooting}
        default{throw 'Unexpected live registry query'}
    }
}
function Require-RequestNonce {param($Request) return $script:Nonce}
function Get-NativeGpuTelemetry {return [pscustomobject]@{uuid='GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29';core_temp_c=31.;power_draw_w=22.;power_limit_w=200.;memory_used_mib=79}}
function Get-MemoryJunctionTemperature {param($Computer) return 38.}
function Get-CpuResourceTelemetry {param($Computer) return [pscustomobject]@{cpu_max_temp_c=29.;cpu_package_power_w=8.}}
function Sign-BridgePayload {param($Rsa,$Payload) return 'synthetic-signature-no-key-access'}
function Write-BridgeResponse {param($Context,$StatusCode,$Body) $script:ResponseStatus=$StatusCode;$script:ResponseBody=$Body}
function Invoke-Case($case,$name,$expected=200,$reuse=$false) {
    $script:Case=$case;$script:Counts=@{memory=0;workers=0;setup=0;registry=0}
    if(-not $reuse){$script:MaintenanceWitnessState=$null;$script:TestJournalRoot=[IO.Path]::Combine($script:Base,$name);$null=[IO.Directory]::CreateDirectory($script:TestJournalRoot)}
    $script:ResponseStatus=0;$script:Nonce=('a'*64)
    $context=[pscustomobject]@{Request=[pscustomobject]@{}}
    $computer=$null;$rsa=$null
    Invoke-Expression $script:Handler
    Assert-Test ($script:ResponseStatus -eq $expected) ($name+' response '+$script:ResponseStatus)
    Assert-Test ($script:Counts.memory -eq 1 -and $script:Counts.workers -in @(0,1) -and $script:Counts.setup -eq 1 -and $script:Counts.registry -le 3) ($name+' multiple source queries')
    if($expected -eq 200){Assert-Test ($script:Counts.workers -eq 1 -and $script:Counts.registry -eq 3) ($name+' missing source snapshot')}
    $script:Passed.Add($name)
}
function Read-Journal {
    $path=[IO.Path]::Combine($script:TestJournalRoot,'GX1-HostTelemetryBridgeMaintenance.jsonl')
    return @([IO.File]::ReadAllLines($path,[Text.UTF8Encoding]::new($false,$true))|ForEach-Object{ConvertFrom-Json -InputObject $_ -ErrorAction Stop})
}
try {
    $idle=New-Case;Invoke-Case $idle 'idle'
    $records=@(Read-Journal)
    Assert-Test ($records.Count -eq 1 -and -not $records[0].maintenance_active -and $records[0].components.workers.Count -eq 0) 'idle witness wrong'
    Assert-Test ($records[0].request_nonce -eq ('a'*64) -and ($records[0].observed_monotonic_ms -is [long] -or $records[0].observed_monotonic_ms -is [int]) -and $records[0].observed_utc.EndsWith('Z')) 'request correlation missing'
    Invoke-Case $idle 'idle-dedup' 200 $true
    Assert-Test (@(Read-Journal).Count -eq 1) 'idle duplicate appended'
    foreach($name in @('TiWorker.exe','MoUsoCoreWorker.exe','SetupHost.exe','SetupPrep.exe','WindowsUpdateBox.exe','wuauclt.exe')){
        $case=New-Case;$case.Workers=@(New-Worker $name);Invoke-Case $case ('worker-'+$name)
        $records=@(Read-Journal);$worker=$records[0].components.workers[0]
        Assert-Test ($records[0].maintenance_active -and $worker.name -ceq $name -and $worker.pid -eq 123 -and $worker.parent_pid -eq 4 -and $worker.creation_utc_known -and $worker.creation_utc -eq '2026-11-27T17:44:40.0000000Z') 'worker cause lost'
        Invoke-Case $case ('active-exact-request-'+$name) 200 $true
        $records=@(Read-Journal)
        Assert-Test ($records.Count -eq 2) 'active exact request was deduplicated'
    }
    foreach($flag in @('setup','oobe','reboot','cbs','wu')){
        $case=New-Case
        switch($flag){'setup'{$case.Setup.SystemSetupInProgress=1};'oobe'{$case.Setup.OOBEInProgress=1};'reboot'{$case.Rebooting=$true};'cbs'{$case.Cbs=$true};'wu'{$case.Wu=$true}}
        Invoke-Case $case ('flag-'+$flag)
        $record=@(Read-Journal)[0]
        Assert-Test ($record.maintenance_active -eq ($flag -in @('setup','oobe','reboot'))) 'flag changed maintenance predicate'
    }
    foreach($bad in @('cbs-null','wu-string','reboot-null','setup-null','setup-2','setup-string','worker-name','worker-pid0','worker-pidnegative','worker-parent','worker-created','worker-duplicate')){
        $case=New-Case;$worker=New-Worker
        switch($bad){
            'cbs-null'{$case.Cbs=$null};'wu-string'{$case.Wu='false'};'reboot-null'{$case.Rebooting=$null}
            'setup-null'{$case.Setup.SystemSetupInProgress=$null};'setup-2'{$case.Setup.SystemSetupInProgress=2};'setup-string'{$case.Setup.SystemSetupInProgress='0'}
            'worker-name'{$worker.Name='not-an-updater.exe';$case.Workers=@($worker)}
            'worker-pid0'{$worker.ProcessId=0;$case.Workers=@($worker)}
            'worker-pidnegative'{$worker.ProcessId=-1;$case.Workers=@($worker)}
            'worker-parent'{$worker.ParentProcessId=-1;$case.Workers=@($worker)}
            'worker-created'{$worker.CreationDate='invalid';$case.Workers=@($worker)}
            'worker-duplicate'{$case.Workers=@($worker,$worker)}
        }
        Invoke-Case $case ('invalid-'+$bad) 503
        Assert-Test ($null -eq $script:MaintenanceWitnessState) 'failed sample advanced state'
    }
    $case=New-Case;$case.Workers=@([pscustomobject]@{Name='TiWorker.exe';ProcessId=[uint32]123});Invoke-Case $case 'optional-worker-times'
    $worker=@(Read-Journal)[0].components.workers[0]
    Assert-Test (-not $worker.creation_utc_known -and $null -eq $worker.creation_utc -and $null -eq $worker.parent_pid) 'missing timestamp invented'
    $case=New-Case;Invoke-Case $case 'transition-idle'
    $before=[IO.File]::ReadAllBytes([IO.Path]::Combine($script:TestJournalRoot,'GX1-HostTelemetryBridgeMaintenance.jsonl'))
    $case.Cbs=$true;Invoke-Case $case 'transition-cbs' 200 $true
    $after=[IO.File]::ReadAllBytes([IO.Path]::Combine($script:TestJournalRoot,'GX1-HostTelemetryBridgeMaintenance.jsonl'))
    Assert-Test (@(Read-Journal).Count -eq 2 -and [Convert]::ToBase64String($after[0..($before.Length-1)]) -ceq [Convert]::ToBase64String($before)) 'append destroyed old records'
    $script:MaintenanceWitnessState=$null;Invoke-Case $case 'service-restart-first-sample' 200 $true
    Assert-Test (@(Read-Journal).Count -eq 3) 'restart erased journal or skipped first sample'
    foreach($failure in @('locked-file','incomplete-record')){
        $script:TestJournalRoot=[IO.Path]::Combine($script:Base,$failure);$null=[IO.Directory]::CreateDirectory($script:TestJournalRoot)
        $path=[IO.Path]::Combine($script:TestJournalRoot,'GX1-HostTelemetryBridgeMaintenance.jsonl')
        $text=if($failure -eq 'locked-file'){'{"preserved":true}'+"`n"}else{'{"partial":'}
        [IO.File]::WriteAllText($path,$text,[Text.UTF8Encoding]::new($false))
        $script:MaintenanceWitnessState=$null;$hold=$null
        try {
            if($failure -eq 'locked-file'){$hold=[IO.FileStream]::new($path,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)}
            Invoke-Case (New-Case) ('io-'+$failure) 503 $true
            Assert-Test ($null -eq $script:MaintenanceWitnessState) 'IO failure advanced state'
        }finally{if($null -ne $hold){$hold.Dispose()}}
        Assert-Test ([IO.File]::ReadAllText($path) -ceq $text) 'IO failure changed old records'
    }
    $hostValue=Get-HostResourceTelemetry
    foreach($bad in @('nonce','monotonic','UTC','signed-disagreement')){
        $candidate=$hostValue.PSObject.Copy();$candidate.maintenance_snapshot=$hostValue.maintenance_snapshot.PSObject.Copy()
        $nonce=('b'*64);$observed=[long]123
        switch($bad){'nonce'{$nonce='invalid'};'monotonic'{$observed=1.5};'UTC'{$candidate.maintenance_snapshot.observed_utc='2026-11-27'};'signed-disagreement'{$candidate.maintenance_active=$true}}
        $failed=$false
        try { $null=Write-MaintenanceWitness -HostResources $candidate -Nonce $nonce -ObservedMonotonicMs $observed -JournalPath ([IO.Path]::Combine($script:Base,'bad.jsonl')) }catch{$failed=$true}
        Assert-Test $failed ('bad correlation accepted '+$bad);$script:Passed.Add('correlation-'+$bad)
    }
    $json=ConvertTo-MaintenanceWitnessJson -Value ([pscustomobject][ordered]@{quoted='a"b';empty=@();none=$null;value=$false})
    Assert-Test ((ConvertFrom-Json $json).quoted -ceq 'a"b') 'strict JSON escaping failed'
    $script:Passed.Add('strict-json')
    [pscustomobject]@{status='PASS_SYNTHETIC_POWERSHELL_MAINTENANCE_WITNESS';cases=@($script:Passed);case_count=$script:Passed.Count;actual_CIM_registry_or_key_queries=0;live_workers_spawned=0;service_or_installer_launched=$false}|ConvertTo-Json -Compress -Depth 5
} finally {
    if([IO.Directory]::Exists($script:Base)){[IO.Directory]::Delete($script:Base,$true)}
}
"""


def test_actual_powershell_snapshot_and_request_handler_with_synthetic_sources(tmp_path):
    if not POWERSHELL.is_file():
        pytest.skip('Windows PowerShell required for actual function runtime test')
    harness = HARNESS.replace('__SERVICE_BASE64__', base64.b64encode(service_source().encode()).decode())
    path = tmp_path / 'maintenance_synthetic.ps1'
    path.write_text(harness, encoding='utf-8')
    windows_path = '\\\\wsl.localhost\\Ubuntu-22.04' + str(path).replace('/', '\\')
    result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', windows_path],
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"status":')]
    assert len(reports) == 1, result.stdout + result.stderr
    report = reports[0]
    assert report['status'] == 'PASS_SYNTHETIC_POWERSHELL_MAINTENANCE_WITNESS'
    assert report['case_count'] >= 35 and len(set(report['cases'])) == report['case_count']
    assert report['actual_CIM_registry_or_key_queries'] == report['live_workers_spawned'] == 0
    assert report['service_or_installer_launched'] is False
