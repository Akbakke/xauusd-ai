# Pure mocked main-loop regression; requires source strings, never loads hardware owners.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$MainSource,[Parameter(Mandatory=$true)][string]$ScopeSource)

Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseInput($mainSource,[ref]$tokens,[ref]$errors)
if ($errors.Count -ne 0) { throw "Main syntax errors: $errors" }
$null=[System.Management.Automation.Language.Parser]::ParseInput($scopeSource,[ref]$tokens,[ref]$errors)
if ($errors.Count -ne 0) { throw "Scope syntax errors: $errors" }
# Only these pure functions from the main owner are loaded. No hardware/service
# implementation from it is ever defined in the test process.
foreach ($name in @('Test-Gx1HighIdleSample','Test-Gx1NormalIdleSample','Invoke-Gx1PolicySelfTest')) {
 $nodes=@($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true))
 if ($nodes.Count -ne 1) { throw "Missing pure owner: $name" }
 . ([scriptblock]::Create($nodes[0].Extent.Text))
}
$policy=Invoke-Gx1PolicySelfTest | ConvertFrom-Json
if ($policy.decision -ne 'PASS') { throw 'Default idle policy regression' }
$body=$mainSource.Substring($mainSource.IndexOf('$benchmarkRequested ='))
$loader=". (Join-Path `$PSScriptRoot 'GX1-PowerBenchmarkScope.ps1')"
if (-not $body.Contains($loader)) { throw 'Exact optional module loader not found' }
# Source-identical control flow; only file loading is supplied in memory. Scope
# functions are reviewed source, GPU/service functions below are test doubles.
$body=$body.Replace($loader,". ([scriptblock]::Create(`$scopeSource))`nfunction Get-Gx1BenchmarkUtcNow { return `$script:clock }`nfunction Get-Gx1BenchmarkMonotonicSeconds { return `$script:monotonic }")
$runBody=[scriptblock]::Create($body)
$testRoot=Join-Path $env:TEMP ('GX1_MAIN_LOOP_TEST_'+[guid]::NewGuid().ToString('N'))
$null=New-Item -ItemType Directory -Path $testRoot
$script:events=@();$script:physical=160;$script:sleeps=0;$script:clock=[datetime]::UtcNow;$script:monotonic=[double]100
function Write-Gx1GuardLog { param($Message) $script:events+=('log:'+ $Message) }
function Stop-Gx1TelemetryBridge { param($Config) $script:events+='stop_bridge' }
function Set-Gx1PowerLimit {
 param($Config)
 $script:physical=[int]$Config.power_limit_w
 $script:events+=('set:'+ $script:physical)
 return [pscustomobject]@{gpu_uuid=$Config.expected_gpu_uuid;power_limit_w=[double]$script:physical;observed_utc=$script:clock.ToString('o')}
}
function Get-Gx1GpuSample {
 param($Config)
 if ($script:scenario -eq 'sample_failure' -and $script:physical -eq 200 -and -not $script:sampleFailed) {
  $script:sampleFailed=$true;throw 'SYNTHETIC_SAMPLE_FAILURE'
 }
 $draw=30.0
 if ($script:scenario -eq 'recovery' -and $script:physical -eq 200) {$draw=95.0}
 return [pscustomobject]@{gpu_uuid=$Config.expected_gpu_uuid;power_limit_w=[double]$script:physical;
  observed_utc=$script:clock.ToString('o');power_draw_w=$draw;memory_used_mib=200;utilization_percent=0;pstate='P0'}
}
function Get-Gx1RecoveryHistory { return @() }
function Save-Gx1RecoveryHistory { param($History) $script:events+='save_recovery_history' }
function Write-Gx1Blocker { param($Sample,$Reason) $script:events+=('blocker:'+ $Reason) }
function Invoke-Gx1GpuRecovery {
 param($Config,$BeforeSample)
 if ($script:physical -ne 160) {throw 'Recovery began above baseline'}
 $script:events+='recovery_at_160'
 return Get-Gx1GpuSample -Config $Config
}
function Start-Sleep {
 param($Seconds)
 $script:sleeps++
 $script:clock=$script:clock.AddSeconds(5);$script:monotonic+=5
 if ($script:scenario -eq 'recovery' -and $script:sleeps -lt 12) {return}
 if ($script:sleeps -eq 1) {
  if ($script:scenario -eq 'close') {
   [System.IO.File]::WriteAllText((Join-Path $script:scopeDirectory 'close.json'),'{"test_only":true}')
   return
  }
  if ($script:scenario -eq 'expiry') {$script:clock=$script:clock.AddSeconds(1800);return}
  if ($script:scenario -eq 'periodic') {return}
 }
 throw 'SYNTHETIC_END_OF_LOOP'
}
$results=@()
foreach ($case in @('default_once','default_loop','close','expiry','sample_failure','finally','periodic','recovery','matched_160','incomplete_scope')) {
 $script:scenario=$case;$script:physical=160;$script:events=@();$script:sleeps=0;$script:sampleFailed=$false
 $script:clock=[datetime]::UtcNow;$script:monotonic=[double]100
 $root=Join-Path $testRoot $case;$null=New-Item -ItemType Directory -Path $root
 $configPath=Join-Path $root 'config.json';$blockerPath=Join-Path $root 'block.json'
 $testConfig=[pscustomobject]@{schema_version='gx1_gpu_power_and_idle_guard_v2';
  expected_gpu_uuid='GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29';gpu_pnp_instance_id='PCI\VEN_10DE&FAKE';
  power_limit_w=160;sample_seconds=5;idle_power_threshold_w=60;idle_memory_max_mib=384;idle_utilization_max_percent=2;
  idle_required_samples=12;max_recoveries_per_window=2;initial_retry_count=1;retry_delay_seconds=1;
  recheck_seconds=900;recovery_window_seconds=3600;recovery_cooldown_seconds=60;telemetry_task_name='FAKE'}
 if ($case -eq 'periodic') {$testConfig.recheck_seconds=0}
 $testConfig | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8
 $Once=($case -eq 'default_once');$PolicySelfTest=$false;$BenchmarkScopePath='';$BenchmarkScopeSha256=''
 if (-not $case.StartsWith('default')) {
  $scopeId=[guid]::NewGuid().ToString();$operatorId=[guid]::NewGuid().ToString()
  $script:scopeDirectory=Join-Path (Join-Path $root 'Benchmarks') $scopeId
  $null=New-Item -ItemType Directory -Path $script:scopeDirectory -Force
  $target=200;if ($case -eq 'matched_160') {$target=160}
  $scope=[pscustomobject]@{schema_version='gx1_operator_power_scope_v2';gpu_uuid=$testConfig.expected_gpu_uuid;
   baseline_power_limit_w=160;target_power_limit_w=$target;draw_stop_w=($target+10);core_stop_c=65;memory_junction_stop_c=80;vram_stop_mib=12288;scope_kind='smoke_comparison';
   profile='smoke';precision_policy='experimental_fp32_3090_no_uninitialized_fill';subsample_rows=512;batch_size=8;grad_accum_steps=1;epochs=1;max_optimizer_steps=64;
   baseline_restore_required=$true;authority=[pscustomobject]@{short_power_comparison=$true;candidate_continuation=$false;test_access=$false;promotion=$false;permanent_power_change=$false};
   scope_id=$scopeId;operator_action_id=$operatorId;created_utc=$script:clock.AddSeconds(-1).ToString('yyyy-MM-ddTHH:mm:ssZ');
   expires_utc=$script:clock.AddSeconds(1790).ToString('yyyy-MM-ddTHH:mm:ssZ');source_commit=('a'*40);recipe_sha256=('b'*64);baseline_reference_report_sha256=('c'*64);run_id='SYNTHETIC_MAIN_LOOP';candidate_gate_sha256=$null;candidate_execution_budget_sha256=$null}
  $BenchmarkScopePath=Join-Path $script:scopeDirectory 'scope.json'
  [System.IO.File]::WriteAllText($BenchmarkScopePath,($scope | ConvertTo-Json -Depth 6),[System.Text.UTF8Encoding]::new($false))
  $BenchmarkScopeSha256=(Get-FileHash $BenchmarkScopePath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($case -eq 'incomplete_scope') {$BenchmarkScopeSha256=''}
 }
 $failure='';$output=@()
 try {$output=@(& $runBody)} catch {$failure=$_.Exception.Message}
 if ($case -eq 'incomplete_scope') {
  if ($failure -notlike 'A benchmark requires*' -or $script:events.Count -ne 0) {throw 'Incomplete scope reached hardware'}
 } elseif ($case -eq 'default_once') {
  if ($failure -ne '' -or ($output[0] | ConvertFrom-Json).decision -ne 'PASS') {throw "Default Once failed: $failure"}
 } elseif ($failure -ne 'SYNTHETIC_END_OF_LOOP') {throw "Unexpected loop termination case=$case failure=$failure"}
 if ($script:physical -ne 160) {throw "Case $case left treatment active"}
 if (-not $case.StartsWith('default') -and $case -ne 'incomplete_scope') {
  $receipt=Get-Content (Join-Path $script:scopeDirectory 'receipt.json') -Raw | ConvertFrom-Json
  if ($receipt.phase -ne 'closed' -or $receipt.baseline_restoration.observed_power_limit_w -ne 160) {throw "No verified closure for $case"}
  if ($case -eq 'recovery' -and $script:events -notcontains 'recovery_at_160') {throw 'Recovery branch not reached'}
  if ($case -eq 'periodic') {
   if (@($script:events | Where-Object {$_ -eq 'set:200'}).Count -ne 1) {throw 'Periodic scope re-ran the slow treatment setter'}
   if (@($script:events | Where-Object {$_ -like 'log:BENCHMARK_TREATMENT_RECHECK_VERIFIED*'}).Count -lt 1) {throw 'Periodic exact sample verification not reached'}
  }
 }
 $results += [pscustomobject]@{case=$case;passed=$true;events=$script:events;physical_final_w=$script:physical}
}
[pscustomobject]@{schema_version='gx1_mocked_native_windows_main_loop_test_v1';passed=$results.Count;cases=$results;default_idle_policy=$policy;
 actual_control_flow=$true;hardware_implementations_loaded=$false;hardware_calls=0;test_root=$testRoot;module_loader='in_memory_reviewed_source';forced_os_kill_tested=$false} | ConvertTo-Json -Depth 9 -Compress
