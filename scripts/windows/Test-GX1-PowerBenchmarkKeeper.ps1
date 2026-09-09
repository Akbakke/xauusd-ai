# Run with the reviewed GX1-PowerBenchmarkScope functions loaded. This suite
# never loads original hardware setters or the installed keeper. Its temporary
# JSON files exercise the real durable token/receipt owner; hardware is fake.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$testRoot = Join-Path $env:TEMP ('GX1_POWER_KEEPER_TEST_' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path (Join-Path $testRoot 'Benchmarks')
$script:now = [datetime]::Parse('2026-09-09T00:01:00Z').ToUniversalTime()
$script:mono = [double]1000
$script:sets = @(); $script:stops = 0; $script:logs = @(); $script:busy = $false
$script:failBaseline = $false; $script:failTarget = $false
function Get-Gx1BenchmarkUtcNow { return $script:now }
function Get-Gx1BenchmarkMonotonicSeconds { return $script:mono }
function Write-Gx1GuardLog { param($Message) $script:logs += $Message }
function Start-Sleep { param($Seconds) }
function Stop-Gx1TelemetryBridge { param($Config) $script:stops += 1 }
function Get-Gx1GpuSample {
    param($Config)
    return [pscustomobject]@{ gpu_uuid=$Config.expected_gpu_uuid;power_limit_w=[double]160;
        observed_utc=$script:now.ToString('o');power_draw_w=[double]30;memory_used_mib=200;utilization_percent=0 }
}
function Test-Gx1NormalIdleSample { param($Sample,$Config) return (-not $script:busy) }
function Set-Gx1PowerLimit {
    param($Config)
    $script:sets += [int]$Config.power_limit_w
    if (($script:failBaseline -and $Config.power_limit_w -eq 160) -or
        ($script:failTarget -and $Config.power_limit_w -eq 200)) { throw 'Synthetic setter failure' }
    return [pscustomobject]@{gpu_uuid=$Config.expected_gpu_uuid;power_limit_w=[double]$Config.power_limit_w;observed_utc=$script:now.ToString('o')}
}
function Assert-Condition { param([bool]$Condition,[string]$Message) if (-not $Condition) { throw $Message } }
$config=[pscustomobject]@{power_limit_w=[int]160;sample_seconds=[int]5;
    expected_gpu_uuid='GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29';telemetry_task_name='FAKE'}
function New-TestScope {
    param([string]$OperatorId='')
    $script:now=[datetime]::Parse('2026-09-09T00:01:00Z').ToUniversalTime();$script:mono=[double]1000
    if ($OperatorId -eq '') { $OperatorId=[guid]::NewGuid().ToString() }
    $scope=[pscustomobject]@{
        schema_version='gx1_operator_power_scope_v2';gpu_uuid=$config.expected_gpu_uuid
        baseline_power_limit_w=160;target_power_limit_w=200;draw_stop_w=210;core_stop_c=65
        memory_junction_stop_c=80;vram_stop_mib=12288;scope_kind='smoke_comparison';profile='smoke'
        precision_policy='experimental_fp32_3090_no_uninitialized_fill';subsample_rows=512;batch_size=8
        grad_accum_steps=1;epochs=1;max_optimizer_steps=64;baseline_restore_required=$true
        authority=[pscustomobject]@{short_power_comparison=$true;candidate_continuation=$false;test_access=$false;promotion=$false;permanent_power_change=$false}
        scope_id=[guid]::NewGuid().ToString();operator_action_id=$OperatorId
        created_utc='2026-09-09T00:00:00Z';expires_utc='2026-09-09T00:30:00Z'
        source_commit=('a'*40);recipe_sha256=('b'*64);baseline_reference_report_sha256=('c'*64);run_id='SYNTHETIC_KEEPER';candidate_gate_sha256=$null;candidate_execution_budget_sha256=$null
    }
    $directory=Join-Path (Join-Path $testRoot 'Benchmarks') $scope.scope_id
    $null=New-Item -ItemType Directory -Path $directory
    $path=Join-Path $directory 'scope.json'
    Write-Gx1BenchmarkJsonDurable -Path $path -Value $scope -Exclusive
    return [pscustomobject]@{Path=$path;Hash=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();Scope=$scope}
}
function New-TestContext { param($InputScope)
    return New-Gx1BenchmarkKeeperContext -ScopePath $InputScope.Path -ScopeSha256 $InputScope.Hash -GuardRoot $testRoot -BaselineConfig $config
}
$checks=@()
$inputScope=New-TestScope;$ctx=New-TestContext $inputScope
Assert-Condition ($ctx.Phase -eq 'arming' -and $ctx.FreshOperatorTokenConsumed -and $script:sets.Count -eq 0) 'Token preparation changed hardware'
Assert-Condition ((Test-Path $ctx.ConsumedPath) -and (Test-Path $ctx.OperatorTokenPath)) 'Tokens not durably written'
$checks+='durable_tokens_before_any_setter'
Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
$receipt=Get-Content $ctx.ReceiptPath -Raw | ConvertFrom-Json
Assert-Condition ($ctx.Phase -eq 'active' -and $script:sets[-1] -eq 200 -and $receipt.phase -eq 'active' -and $receipt.scope_sha256 -ceq $inputScope.Hash) 'Arming/receipt mismatch'
$checks+='first_active_receipt_binds_exact_scope'
$before=$script:sets.Count;$previousReceiptTime=$receipt.observed_utc;$script:now=$script:now.AddSeconds(5)
Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
$heartbeat=Get-Content $ctx.ReceiptPath -Raw | ConvertFrom-Json
Assert-Condition ($heartbeat.observed_utc -cne $previousReceiptTime -and $heartbeat.phase -eq 'active') 'Active poll did not renew keeper heartbeat'
$checks+='active_receipt_heartbeat_advances_without_new_operator_token'
Assert-Condition ($script:sets.Count -eq $before) 'Every poll reapplied target power' 
$checks+='active_poll_does_not_repeat_setter'
Close-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config -Reason completed
$receipt=Get-Content $ctx.ReceiptPath -Raw | ConvertFrom-Json
Assert-Condition ($ctx.Phase -eq 'closed' -and (Test-Path $ctx.ClosePath) -and $receipt.phase -eq 'closed' -and $receipt.baseline_restoration.observed_power_limit_w -eq 160) 'Closure not durably verified'
$checks+='normal_close_has_physical_baseline_receipt'
$before=$script:sets.Count;Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
Assert-Condition ($script:sets.Count -eq $before) 'Closed scope was reused'
$checks+='closed_context_cannot_rearm'
foreach ($fault in @('restart','utc_expiry','monotonic_expiry','caller_close','source_changed','scope_token_changed','operator_token_changed')) {
    $inputScope=New-TestScope;$ctx=New-TestContext $inputScope
    Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
    $targetCount=@($script:sets | Where-Object { $_ -eq 200 }).Count
    switch ($fault) {
        'restart' { $ctx=New-TestContext $inputScope;Assert-Condition ($ctx.Phase -eq 'closing') 'Restart replayed authorization' }
        'utc_expiry' { $script:now=$script:now.AddMinutes(30) }
        'monotonic_expiry' { $script:mono += 1800 }
        'caller_close' { Write-Gx1BenchmarkJsonDurable -Path $ctx.ClosePath -Value ([pscustomobject]@{closed=$true}) -Exclusive }
        'source_changed' { [IO.File]::AppendAllText($ctx.ScopePath,' ') }
        'scope_token_changed' { [IO.File]::AppendAllText($ctx.ConsumedPath,' ') }
        'operator_token_changed' { [IO.File]::AppendAllText($ctx.OperatorTokenPath,' ') }
    }
    Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
    Assert-Condition ($ctx.Phase -eq 'closed' -and $script:sets[-1] -eq 160 -and @($script:sets | Where-Object { $_ -eq 200 }).Count -eq $targetCount) "Fault did not close baseline-only: $fault"
    $checks += "closes_$fault"
}
$inputScope=New-TestScope;$ctx=New-TestContext $inputScope
Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
$script:failBaseline=$true;$failed=$false
try { Close-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config -Reason expired } catch { $failed=$true }
$receipt=Get-Content $ctx.ReceiptPath -Raw | ConvertFrom-Json
Assert-Condition ($failed -and $ctx.Phase -eq 'closing' -and $receipt.phase -eq 'closing' -and $script:stops -ge 3 -and (Test-Path $ctx.ClosePath)) 'Failed restore claimed success or left scope open'
$checks+='failed_restore_keeps_durable_closure_and_stops_bridge'
$targetCount=@($script:sets | Where-Object { $_ -eq 200 }).Count;$script:failBaseline=$false
Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
Assert-Condition ($ctx.Phase -eq 'closed' -and @($script:sets | Where-Object { $_ -eq 200 }).Count -eq $targetCount) 'Restore retry rearmed target'
$checks+='later_restore_retry_is_baseline_only'
$inputScope=New-TestScope;$ctx=New-TestContext $inputScope;$script:failTarget=$true;$failed=$false
try { Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config } catch { $failed=$true }
Assert-Condition ($failed -and $ctx.Phase -eq 'active') 'Setter failure lost context before outer catch'
$script:failTarget=$false;Close-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config -Reason guard_failure
Assert-Condition ($ctx.Phase -eq 'closed' -and $script:sets[-1] -eq 160) 'Arming failure not restorable'
$checks+='arming_failure_retains_context_for_outer_guard_close'
$inputScope=New-TestScope;$script:busy=$true;$before=$script:sets.Count;$failed=$false
try { $null=New-TestContext $inputScope } catch { $failed=$true }
$script:busy=$false
Assert-Condition ($failed -and $script:sets.Count -eq $before -and -not (Test-Path (Join-Path (Split-Path $inputScope.Path) 'consumed.json'))) 'Busy GPU reached token/hardware action'
$checks+='busy_gpu_rejected_before_token_consumption'
$first=New-TestScope;$ctx=New-TestContext $first;$second=New-TestScope -OperatorId $first.Scope.operator_action_id
$secondCtx=New-TestContext $second
Assert-Condition ($secondCtx.Phase -eq 'closing' -and -not $secondCtx.FreshOperatorTokenConsumed) 'Operator token reused across scopes'
$checks+='operator_action_cannot_authorize_two_scope_ids'
$inputScope=New-TestScope
$inputScope.Scope.scope_kind='candidate_continuation'
$inputScope.Scope.profile='candidate';$inputScope.Scope.subsample_rows=[int]0
$inputScope.Scope.epochs=[int]30;$inputScope.Scope.max_optimizer_steps=[int]0
$inputScope.Scope.authority.short_power_comparison=$false
$inputScope.Scope.authority.candidate_continuation=$true
$inputScope.Scope.candidate_gate_sha256=('d'*64)
$inputScope.Scope.candidate_execution_budget_sha256=('e'*64)
$inputScope.Scope.expires_utc='2026-09-09T01:40:00Z'
[IO.File]::Delete($inputScope.Path)
Write-Gx1BenchmarkJsonDurable -Path $inputScope.Path -Value $inputScope.Scope -Exclusive
$inputScope.Hash=(Get-FileHash -LiteralPath $inputScope.Path -Algorithm SHA256).Hash.ToLowerInvariant()
$ctx=New-TestContext $inputScope
Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
Assert-Condition ($ctx.Phase -eq 'active' -and $script:sets[-1] -eq 200) 'Candidate continuation scope did not arm'
Close-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config -Reason completed
Assert-Condition ($ctx.Phase -eq 'closed' -and $script:sets[-1] -eq 160) 'Candidate continuation scope did not restore'
$checks+='candidate_continuation_has_bounded_single_use_lifecycle'
$inputScope=New-TestScope;$config.sample_seconds=[int]6;$failed=$false
try { $null=New-TestContext $inputScope } catch { $failed=$true }
$config.sample_seconds=[int]5
Assert-Condition $failed 'Slow polling allowed an unbounded restore delay'
$checks+='polling_slower_than_five_seconds_rejected'
$inputScope=New-TestScope
$inputScope.Scope.target_power_limit_w=[int]160;$inputScope.Scope.draw_stop_w=[int]170
Write-Gx1BenchmarkJsonDurable -Path $inputScope.Path -Value $inputScope.Scope
$inputScope.Hash=(Get-FileHash -LiteralPath $inputScope.Path -Algorithm SHA256).Hash.ToLowerInvariant()
$before=$script:sets.Count;$ctx=New-TestContext $inputScope
Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
$referenceReceipt=Get-Content $ctx.ReceiptPath -Raw | ConvertFrom-Json
Assert-Condition ($ctx.Phase -eq 'active' -and $script:sets[-1] -eq 160 -and $referenceReceipt.requested_power_limit_w -eq 160) 'Reference point did not use identical scope lifecycle at 160 W'
$script:now=$script:now.AddSeconds(5);Sync-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config
Assert-Condition ($script:sets.Count -eq $before+1) 'Reference heartbeat repeatedly set power'
Close-Gx1BenchmarkKeeperContext -Context $ctx -BaselineConfig $config -Reason completed
Assert-Condition ($ctx.Phase -eq 'closed' -and $script:sets[-1] -eq 160) 'Reference closure failed'
$checks+='matched_160w_reference_has_same_token_receipt_heartbeat_and_closure_path'
[pscustomobject]@{decision='PASS_NATIVE_KEEPER_CONTROLLER_WITH_MOCK_HARDWARE';passed=$checks.Count;checks=$checks;temporary_evidence_root=$testRoot;
    real_durable_files_tested=$true;physical_gpu_calls=0;real_service_calls=0;actual_installed_keeper_loop_tested=$false} | ConvertTo-Json -Depth 6 -Compress
