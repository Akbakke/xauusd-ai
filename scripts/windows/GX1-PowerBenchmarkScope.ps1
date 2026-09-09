# Source-only preparation. Defining these functions does not read files, arm a
# scope or change hardware. The Linux guard independently verifies source and
# recipe content; this parser verifies the identical first-point JSON policy.

function ConvertFrom-Gx1PowerBenchmarkScope {
    param(
        [Parameter(Mandatory = $true)][psobject]$Value,
        [Parameter(Mandatory = $true)][datetime]$NowUtc,
        [Parameter(Mandatory = $true)][string]$ExpectedSourceCommit,
        [Parameter(Mandatory = $true)][string]$ExpectedRecipeSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedBaselineReportSha256
    )
    if ($NowUtc.Kind -ne [DateTimeKind]::Utc) { throw 'Verified current UTC time required' }
    $fixed = [ordered]@{
        schema_version = 'gx1_operator_power_benchmark_scope_draft_v1'
        gpu_uuid = 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
        baseline_power_limit_w = 160; target_power_limit_w = 200
        draw_stop_w = 210; core_stop_c = 65; memory_junction_stop_c = 80
        vram_stop_mib = 12288; profile = 'smoke'
        precision_policy = 'experimental_fp32_3090_no_uninitialized_fill'
        subsample_rows = 512; batch_size = 8; grad_accum_steps = 1
        epochs = 1; max_optimizer_steps = 64; baseline_restore_required = $true
    }
    $variable = @('authority', 'scope_id', 'operator_action_id', 'created_utc', 'expires_utc',
        'source_commit', 'recipe_sha256', 'baseline_reference_report_sha256', 'run_id')
    $expectedKeys = @($fixed.Keys) + $variable
    $actualKeys = @($Value.PSObject.Properties.Name)
    if ($actualKeys.Count -ne $expectedKeys.Count -or
        @($actualKeys | Where-Object { $expectedKeys -cnotcontains $_ }).Count -ne 0) {
        throw 'Power scope fields differ from the first-point contract'
    }
    if ($Value.target_power_limit_w -isnot [int] -or $Value.target_power_limit_w -notin @(160,200)) {
        throw 'Only matched 160 W reference or 200 W treatment is allowed'
    }
    $fixed['target_power_limit_w'] = $Value.target_power_limit_w
    $fixed['draw_stop_w'] = $Value.target_power_limit_w + 10
    foreach ($key in $fixed.Keys) {
        $expected = $fixed[$key]; $actual = $Value.$key
        if ($null -eq $actual -or $actual.GetType() -ne $expected.GetType() -or $actual -cne $expected) {
            throw "Power scope fixed value differs: $key"
        }
    }
    $authority = [ordered]@{
        short_power_comparison = $true; candidate_continuation = $false
        test_access = $false; promotion = $false; permanent_power_change = $false
    }
    $actualAuthorityKeys = @($Value.authority.PSObject.Properties.Name)
    if ($actualAuthorityKeys.Count -ne $authority.Count -or
        @($actualAuthorityKeys | Where-Object { @($authority.Keys) -cnotcontains $_ }).Count -ne 0) {
        throw 'Power authority fields differ'
    }
    foreach ($key in $authority.Keys) {
        if ($Value.authority.$key -isnot [bool] -or $Value.authority.$key -ne $authority[$key]) {
            throw "Power authority value differs: $key"
        }
    }
    foreach ($key in @('scope_id', 'operator_action_id')) {
        if ($Value.$key -isnot [string] -or
            $Value.$key -cnotmatch '\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\z') {
            throw 'Explicit canonical scope/operator action identity required'
        }
    }
    if ($Value.scope_id -ceq $Value.operator_action_id) { throw 'Scope is not operator action identity' }
    foreach ($binding in @(
        @('source_commit', 40, $ExpectedSourceCommit),
        @('recipe_sha256', 64, $ExpectedRecipeSha256),
        @('baseline_reference_report_sha256', 64, $ExpectedBaselineReportSha256)
    )) {
        $key = [string]$binding[0]; $length = [int]$binding[1]; $expected = [string]$binding[2]
        if ($Value.$key -isnot [string] -or $Value.$key -cnotmatch "\A[0-9a-f]{$length}\z" -or
            $Value.$key -cne $expected) { throw "Power benchmark binding mismatch: $key" }
    }
    if ($Value.run_id -isnot [string] -or $Value.run_id -cnotmatch '\A[A-Z0-9_]{1,128}\z') {
        throw 'Exact benchmark run identity required'
    }
    $timestamps = @{}
    foreach ($key in @('created_utc', 'expires_utc')) {
        if ($Value.$key -isnot [string] -or
            $Value.$key -cnotmatch '\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\z') {
            throw 'Canonical UTC timestamp required'
        }
        $timestamps[$key] = [datetime]::ParseExact($Value.$key, "yyyy-MM-dd'T'HH:mm:ss'Z'",
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::AssumeUniversal -bor [Globalization.DateTimeStyles]::AdjustToUniversal)
    }
    $duration = ($timestamps.expires_utc - $timestamps.created_utc).TotalSeconds
    if ($duration -le 0 -or $duration -gt 1800 -or
        $timestamps.created_utc -gt $NowUtc.AddSeconds(2) -or $NowUtc -ge $timestamps.expires_utc) {
        throw 'Power scope is future-dated, expired or exceeds the 30-minute bound'
    }
    return [pscustomobject]@{
        scope = ($Value | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json)
        validation_only = $true; operator_action_verified = $false
        physical_power_changed = $false; source_and_recipe_content_verified = $false
        linux_guard_integrated = $false; windows_keeper_integrated = $false
    }
}

function Assert-Gx1ExactBenchmarkTreatment {
    param([Parameter(Mandatory = $true)][psobject]$Scope,
          [Parameter(Mandatory = $true)][psobject]$Sample)
    if ($Sample.gpu_uuid -cne $Scope.gpu_uuid -or
        ($Sample.power_limit_w -isnot [double] -and $Sample.power_limit_w -isnot [int]) -or
        [double]::IsNaN([double]$Sample.power_limit_w) -or
        [double]::IsInfinity([double]$Sample.power_limit_w) -or
        $Sample.power_limit_w -ne $Scope.target_power_limit_w) {
        throw 'Observed configured watts/UUID do not match the operator treatment'
    }
}

function Restore-Gx1BenchmarkBaseline {
    param(
        [Parameter(Mandatory = $true)][psobject]$BaselineConfig,
        [Parameter(Mandatory = $true)][string]$ScopeId,
        [Parameter(Mandatory = $true)][string]$Reason
    )

    # This helper closes the physical treatment only. Its caller must first
    # durably close the authorization, preventing reactivation after recovery.
    # Never use a candidate config here: recovery always gets the baseline.
    if ($ScopeId -cnotmatch '\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\z' -or
        $Reason -notin @('completed', 'expired', 'cancelled', 'telemetry_failure',
                         'guard_failure', 'recovery', 'authorization_failure')) {
        throw 'benchmark restoration identity/reason invalid'
    }
    if ($BaselineConfig.power_limit_w -isnot [int] -or
        $BaselineConfig.power_limit_w -ne 160) {
        throw 'benchmark restoration requires the unchanged integer 160 W baseline'
    }

    $lastFailure = ''
    foreach ($attempt in 1..3) {
        try {
            $sample = Set-Gx1PowerLimit -Config $BaselineConfig
            # Set-Gx1PowerLimit historically proves a ceiling. Closure needs
            # exact treatment identity; a lower limit is not a 160 W receipt.
            if ($sample.gpu_uuid -cne [string]$BaselineConfig.expected_gpu_uuid -or
                $sample.power_limit_w -isnot [double] -or
                [double]::IsNaN($sample.power_limit_w) -or
                [double]::IsInfinity($sample.power_limit_w) -or
                $sample.power_limit_w -ne 160.0) {
                throw 'benchmark baseline restoration observation mismatch'
            }
            Write-Gx1GuardLog "BENCHMARK_BASELINE_RESTORED scope_id=$ScopeId reason=$Reason power_limit_w=160 attempt=$attempt"
            return [pscustomobject]@{
                schema_version = 'gx1_power_benchmark_baseline_restoration_v1'
                decision = 'PASS_BASELINE_PHYSICALLY_RESTORED'
                scope_id = $ScopeId
                reason = $Reason
                gpu_uuid = $sample.gpu_uuid
                observed_power_limit_w = $sample.power_limit_w
                attempts = $attempt
                telemetry_stop_attempted = ($attempt -gt 1)
                learning_resume_authorized = $false
                observed_utc = $sample.observed_utc
            }
        }
        catch {
            $lastFailure = $_.Exception.Message
            # An unknown/high physical limit must stop guarded training now,
            # even if a subsequent restoration attempt succeeds.
            try {
                Stop-Gx1TelemetryBridge -Config $BaselineConfig
            }
            catch {
                $lastFailure += "; telemetry_stop_failure=$($_.Exception.Message)"
            }
            Write-Gx1GuardLog "BENCHMARK_RESTORE_FAILURE scope_id=$ScopeId reason=$Reason attempt=$attempt message=$lastFailure"
            if ($attempt -lt 3) { Start-Sleep -Seconds 1 }
        }
    }
    # A failed restore never creates a successful closure receipt. The caller
    # must retain its closed scope/blocker and retry baseline enforcement only.
    throw "benchmark baseline restoration unproven: $lastFailure"
}


function Get-Gx1BenchmarkLoopDecision {
    # Pure state transition. The keeper must perform durable token consumption,
    # closure and physical verification; booleans here are observations, never
    # caller-facing overrides. No result itself sets hardware or starts training.
    param(
        [Parameter(Mandatory = $true)][ValidateSet('baseline','arming','active','closing','closed')][string]$Phase,
        [ValidateSet(160,200)][int]$TargetPowerLimit = 200,
        [bool]$ScopeValidated = $false,
        [bool]$FreshOperatorTokenConsumed = $false,
        [bool]$KeeperRestarted = $false,
        [bool]$Expired = $false,
        [bool]$ClosureRequested = $false,
        [bool]$ScopeChanged = $false,
        [bool]$RecoveryRequested = $false,
        [bool]$SampleFailed = $false,
        [bool]$TreatmentMismatch = $false,
        [bool]$BaselineRestorationVerified = $false
    )
    $reason = $null
    foreach ($item in @(
        @($KeeperRestarted,'keeper_restarted'), @($Expired,'expired'),
        @($ClosureRequested,'closed_by_caller'), @($ScopeChanged,'scope_changed'),
        @($RecoveryRequested,'recovery'), @($SampleFailed,'sample_failure'),
        @($TreatmentMismatch,'treatment_mismatch')
    )) {
        if ([bool]$item[0]) { $reason = [string]$item[1]; break }
    }
    $next = $Phase
    if ($Phase -eq 'closed') { $next = 'closed' }
    elseif ($Phase -eq 'closing') {
        if ($BaselineRestorationVerified) { $next = 'closed' }
    }
    elseif ($Phase -eq 'baseline') {
        # A fresh operator action enters 'arming' only after durable exclusive
        # token consumption in the owner. Polling a request cannot auto-arm.
        $next = 'baseline'
    }
    elseif ($null -ne $reason -or -not $ScopeValidated -or -not $FreshOperatorTokenConsumed) {
        $next = 'closing'
        if ($null -eq $reason) { $reason = 'authorization_invalid' }
    }
    elseif ($Phase -eq 'arming') { $next = 'active' }
    return [pscustomobject]@{
        previous_phase = $Phase; next_phase = $next; reason = $reason
        requested_power_limit_w = $(if ($next -eq 'active') { $TargetPowerLimit } else { 160 })
        must_durably_close_before_restore = ($next -eq 'closing')
        restoration_required = ($next -eq 'closing')
        physical_closure_proven = ($next -eq 'closed' -and $BaselineRestorationVerified)
        may_reuse_consumed_operator_token = $false
        physical_power_changed = $false
        training_launch_authority = $false
    }
}

function Write-Gx1BenchmarkJsonDurable {
    param([Parameter(Mandatory = $true)][string]$Path,
          [Parameter(Mandatory = $true)][psobject]$Value,
          [switch]$Exclusive)
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 12 -Compress) + "`n")
    $destination = if ($Exclusive) { $Path } else { $Path + '.tmp.' + [guid]::NewGuid().ToString('N') }
    $stream = [IO.FileStream]::new($destination, [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write, [IO.FileShare]::Read, 4096, [IO.FileOptions]::WriteThrough)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
    if (-not $Exclusive) {
        if ([IO.File]::Exists($Path)) { [IO.File]::Replace($destination, $Path, [System.Management.Automation.Language.NullString]::Value) }
        else { [IO.File]::Move($destination, $Path) }
    }
}

function Get-Gx1BenchmarkUtcNow { return [datetime]::UtcNow }
function Get-Gx1BenchmarkMonotonicSeconds {
    return [Diagnostics.Stopwatch]::GetTimestamp() / [double][Diagnostics.Stopwatch]::Frequency
}

function Write-Gx1BenchmarkReceipt {
    param([Parameter(Mandatory = $true)][psobject]$Context,
          [Parameter(Mandatory = $true)][string]$Phase,
          [string]$Reason = '', [psobject]$Restoration = $null)
    Write-Gx1BenchmarkJsonDurable -Path $Context.ReceiptPath -Value ([pscustomobject]@{
        schema_version = 'gx1_operator_power_benchmark_keeper_receipt_draft_v1'
        scope_id = $Context.Scope.scope_id; operator_action_id = $Context.Scope.operator_action_id
        scope_sha256 = $Context.ScopeSha256; gpu_uuid = $Context.Scope.gpu_uuid
        source_commit = $Context.Scope.source_commit; recipe_sha256 = $Context.Scope.recipe_sha256
        run_id = $Context.Scope.run_id; phase = $Phase; reason = $Reason
        expires_utc = $Context.Scope.expires_utc
        observed_utc = (Get-Gx1BenchmarkUtcNow).ToString('o')
        keeper_pid = $PID; requested_power_limit_w = $(if ($Phase -eq 'active') { $Context.Scope.target_power_limit_w } else { 160 })
        baseline_restoration = $Restoration; training_launch_authority = $false
    })
}

function New-Gx1BenchmarkKeeperContext {
    param([Parameter(Mandatory = $true)][string]$ScopePath,
          [Parameter(Mandatory = $true)][string]$ScopeSha256,
          [Parameter(Mandatory = $true)][string]$GuardRoot,
          [Parameter(Mandatory = $true)][psobject]$BaselineConfig)
    if ($BaselineConfig.power_limit_w -isnot [int] -or $BaselineConfig.power_limit_w -ne 160 -or
        $ScopeSha256 -cnotmatch '\A[0-9a-f]{64}\z' -or
        $BaselineConfig.sample_seconds -isnot [int] -or $BaselineConfig.sample_seconds -lt 1 -or
        $BaselineConfig.sample_seconds -gt 5) { throw 'Exact baseline, scope digest and at-most-five-second polling required' }
    if (-not [IO.Path]::IsPathRooted($ScopePath) -or
        -not (Test-Path -LiteralPath $ScopePath -PathType Leaf) -or
        ((Get-Item -LiteralPath $ScopePath).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Absolute regular operator scope file required'
    }
    if ((Get-FileHash -LiteralPath $ScopePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ScopeSha256) {
        throw 'Operator scope bytes changed'
    }
    $scope = Get-Content -LiteralPath $ScopePath -Raw -Encoding UTF8 | ConvertFrom-Json
    # Windows verifies declared fields, not Linux repository contents. The
    # source-bound Linux guard must independently match the actual invocation.
    $validated = ConvertFrom-Gx1PowerBenchmarkScope -Value $scope -NowUtc (Get-Gx1BenchmarkUtcNow) `
        -ExpectedSourceCommit $scope.source_commit -ExpectedRecipeSha256 $scope.recipe_sha256 `
        -ExpectedBaselineReportSha256 $scope.baseline_reference_report_sha256
    $scope = $validated.scope
    $benchmarkRoot = Join-Path $GuardRoot 'Benchmarks'
    $directory = Join-Path $benchmarkRoot $scope.scope_id
    $expectedPath = Join-Path $directory 'scope.json'
    if (-not [string]::Equals([IO.Path]::GetFullPath($ScopePath), [IO.Path]::GetFullPath($expectedPath),
        [StringComparison]::OrdinalIgnoreCase)) { throw 'Scope must occupy its exact operator directory' }
    foreach ($path in @($GuardRoot, $benchmarkRoot, $directory)) {
        if (((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Operator scope directory cannot be a reparse point'
        }
    }
    $context = [pscustomobject]@{
        Scope = $scope; ScopePath = $ScopePath; ScopeSha256 = $ScopeSha256
        ReceiptPath = (Join-Path $directory 'receipt.json')
        ClosePath = (Join-Path $directory 'close.json')
        ConsumedPath = (Join-Path $directory 'consumed.json')
        OperatorTokenPath = (Join-Path $benchmarkRoot ('operator-' + $scope.operator_action_id + '.consumed.json'))
        Phase = 'arming'; FreshOperatorTokenConsumed = $false; ConsumedTokenSha256 = ''
        StartedMonotonicSeconds = (Get-Gx1BenchmarkMonotonicSeconds)
        MaximumRemainingSeconds = ([datetime]::Parse($scope.expires_utc).ToUniversalTime() - (Get-Gx1BenchmarkUtcNow)).TotalSeconds
        PowerConfig = ($BaselineConfig | ConvertTo-Json -Depth 8 | ConvertFrom-Json)
    }
    $context.PowerConfig.power_limit_w = [int]$scope.target_power_limit_w
    # A keeper restart or duplicate operator action cannot replay a consumed
    # token, even if the original JSON and its expiry remain valid.
    if ((Test-Path -LiteralPath $context.ConsumedPath) -or
        (Test-Path -LiteralPath $context.OperatorTokenPath) -or
        (Test-Path -LiteralPath $context.ClosePath)) {
        $context.Phase = 'closing'
        return $context
    }
    $sample = Get-Gx1GpuSample -Config $BaselineConfig
    if ($sample.gpu_uuid -cne $scope.gpu_uuid -or $sample.power_limit_w -ne 160 -or
        -not (Test-Gx1NormalIdleSample -Sample $sample -Config $BaselineConfig)) {
        throw 'Benchmark requires verified baseline and idle GPU before token consumption'
    }
    $token = [pscustomobject]@{
        schema_version = 'gx1_power_benchmark_consumed_operator_token_draft_v1'
        scope_id = $scope.scope_id; operator_action_id = $scope.operator_action_id
        scope_sha256 = $ScopeSha256; keeper_pid = $PID
        consumed_utc = (Get-Gx1BenchmarkUtcNow).ToString('o'); reusable = $false
    }
    try {
        Write-Gx1BenchmarkJsonDurable -Path $context.OperatorTokenPath -Value $token -Exclusive
        Write-Gx1BenchmarkJsonDurable -Path $context.ConsumedPath -Value $token -Exclusive
        $context.ConsumedTokenSha256 = (Get-FileHash -LiteralPath $context.ConsumedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $context.FreshOperatorTokenConsumed = $true
    }
    catch {
        $context.Phase = 'closing'
        Write-Gx1GuardLog "BENCHMARK_TOKEN_FAILURE scope_id=$($scope.scope_id) message=$($_.Exception.Message)"
    }
    return $context
}

function Close-Gx1BenchmarkKeeperContext {
    param([Parameter(Mandatory = $true)][psobject]$Context,
          [Parameter(Mandatory = $true)][psobject]$BaselineConfig,
          [Parameter(Mandatory = $true)][ValidateSet('completed','expired','cancelled','telemetry_failure','guard_failure','recovery','authorization_failure')][string]$Reason)
    if ($Context.Phase -eq 'closed') { return }
    $Context.Phase = 'closing'
    $Context.FreshOperatorTokenConsumed = $false
    $closureFailure = $null
    try {
        if (-not (Test-Path -LiteralPath $Context.ClosePath)) {
            Write-Gx1BenchmarkJsonDurable -Path $Context.ClosePath -Exclusive -Value ([pscustomobject]@{
                scope_id = $Context.Scope.scope_id; scope_sha256 = $Context.ScopeSha256
                reason = $Reason; closed_utc = (Get-Gx1BenchmarkUtcNow).ToString('o')
            })
        }
        Write-Gx1BenchmarkReceipt -Context $Context -Phase closing -Reason $Reason
    }
    catch {
        $closureFailure = $_.Exception.Message
        # If durable closure failed, stop the signed bridge before restoring.
        # Consumed startup tokens still prohibit rearming on process restart.
        try { Stop-Gx1TelemetryBridge -Config $BaselineConfig } catch { }
    }
    # Always attempt restoration, including after a disk/receipt failure.
    $restoration = Restore-Gx1BenchmarkBaseline -BaselineConfig $BaselineConfig `
        -ScopeId $Context.Scope.scope_id -Reason $Reason
    if ($null -ne $closureFailure) { throw "Baseline restored but closure persistence failed: $closureFailure" }
    Write-Gx1BenchmarkReceipt -Context $Context -Phase closed -Reason $Reason -Restoration $restoration
    $Context.Phase = 'closed'
}

function Sync-Gx1BenchmarkKeeperContext {
    param([Parameter(Mandatory = $true)][psobject]$Context,
          [Parameter(Mandatory = $true)][psobject]$BaselineConfig)
    if ($Context.Phase -eq 'closed') { return }
    if ($Context.Phase -eq 'closing') {
        Close-Gx1BenchmarkKeeperContext -Context $Context -BaselineConfig $BaselineConfig -Reason authorization_failure
        return
    }
    $changed = $true
    if (Test-Path -LiteralPath $Context.ScopePath -PathType Leaf) {
        $changed = (Get-FileHash -LiteralPath $Context.ScopePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Context.ScopeSha256
    }
    foreach ($tokenPath in @($Context.ConsumedPath, $Context.OperatorTokenPath)) {
        if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf) -or
            (Get-FileHash -LiteralPath $tokenPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Context.ConsumedTokenSha256) {
            $changed = $true
        }
    }
    $expired = ((Get-Gx1BenchmarkUtcNow) -ge [datetime]::Parse($Context.Scope.expires_utc).ToUniversalTime()) -or
        ((Get-Gx1BenchmarkMonotonicSeconds) - $Context.StartedMonotonicSeconds -ge $Context.MaximumRemainingSeconds)
    $decision = Get-Gx1BenchmarkLoopDecision -Phase $Context.Phase -TargetPowerLimit $Context.Scope.target_power_limit_w -ScopeValidated (-not $changed) `
        -FreshOperatorTokenConsumed $Context.FreshOperatorTokenConsumed -Expired $expired `
        -ClosureRequested (Test-Path -LiteralPath $Context.ClosePath) -ScopeChanged $changed
    if ($decision.next_phase -eq 'closing') {
        $reason = if ($expired) { 'expired' } elseif ($changed) { 'authorization_failure' } else { 'completed' }
        Close-Gx1BenchmarkKeeperContext -Context $Context -BaselineConfig $BaselineConfig -Reason $reason
        return
    }
    if ($Context.Phase -eq 'arming') {
        # Context already exists in the main loop before a setter can fail.
        $Context.Phase = 'active'
        $sample = Set-Gx1PowerLimit -Config $Context.PowerConfig
        Assert-Gx1ExactBenchmarkTreatment -Scope $Context.Scope -Sample $sample
        Write-Gx1BenchmarkReceipt -Context $Context -Phase active
        Write-Gx1GuardLog "BENCHMARK_ACTIVE scope_id=$($Context.Scope.scope_id) power_limit_w=$($Context.Scope.target_power_limit_w) expires_utc=$($Context.Scope.expires_utc)"
    }
    elseif ($Context.Phase -eq 'active') {
        # A fresh receipt lets the independent Linux guard detect a keeper
        # crash even while the signed hardware telemetry bridge remains alive.
        Write-Gx1BenchmarkReceipt -Context $Context -Phase active
    }
}
