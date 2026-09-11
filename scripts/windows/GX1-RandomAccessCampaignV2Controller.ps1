param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanFileSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][Alias('WindowsSourceRepo')][string]$WindowsControllerSourceRoot,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$Python = '.venv/bin/python'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Invoke-Gx1Json {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [ValidateRange(1, 30000)][int]$TimeoutMilliseconds = 30000
    )
    $result = Invoke-Gx1WslBounded -Arguments (@(
        '--cd', $SourceRepo, '--', $Python, '-m', 'gx1.scripts.local_random_access_campaign_v2'
    ) + $Arguments) -TimeoutMilliseconds $TimeoutMilliseconds
    $lines = @($result.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
    if ($result.ExitCode -ne 0 -or $lines.Count -ne 1) {
        throw "Random-access campaign command failed: $($result.StdErr.Trim())"
    }
    $value = $lines[0] | ConvertFrom-Json
    if ($value.ok -ne $true) { throw 'Random-access campaign command returned non-PASS' }
    return $value
}
function Convert-Gx1WslPath {
    param([Parameter(Mandatory = $true)][string]$LinuxPath)
    $result = Invoke-Gx1WslBounded -Arguments @('--', 'wslpath', '-w', $LinuxPath) -TimeoutMilliseconds 10000
    $value = @($result.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
    if ($result.ExitCode -ne 0 -or $value.Count -ne 1) {
        throw "WSL path conversion failed: $($result.StdErr.Trim())"
    }
    return $value[0]
}
function Write-Gx1BootIdentity {
    param([ValidateRange(1, 10000)][int]$WslTimeoutMilliseconds = 10000)
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $bootId = (Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters' -Name BootId -ErrorAction Stop).BootId
    $bootUtc = $operatingSystem.LastBootUpTime.ToUniversalTime().ToString('o')
    $computer = [Environment]::MachineName
    $unsigned = '{"boot_id":' + [string]$bootId + ',"computer_name":"' + $computer + '","last_boot_utc":"' + $bootUtc + '","schema_version":"gx1_windows_boot_identity_v1"}' + "`n"
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($unsigned)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $digestBytes = $sha.ComputeHash($bytes) } finally { $sha.Dispose() }
    $digest = ([BitConverter]::ToString($digestBytes) -replace '-', '').ToLowerInvariant()
    $root = Join-Path $env:ProgramData 'GX1\RandomAccessCampaignV2'
    New-Item -ItemType Directory -Path $root -Force | Out-Null
    $path = Join-Path $root 'CURRENT_BOOT.json'
    $value = [ordered]@{
        schema_version = 'gx1_windows_boot_identity_v1'
        computer_name = $computer
        last_boot_utc = $bootUtc
        boot_id = [int64]$bootId
        identity_sha256 = $digest
    }
    $temporary = $path + '.tmp.' + [guid]::NewGuid().ToString('N')
    $backup = $path + '.backup.' + [guid]::NewGuid().ToString('N')
    try {
        [IO.File]::WriteAllText($temporary, (($value | ConvertTo-Json -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $path) {
            [IO.File]::Replace($temporary, $path, $backup)
            Remove-Item -LiteralPath $backup -Force
        }
        else {
            [IO.File]::Move($temporary, $path)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
        if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Force }
    }
    # wsl.exe consumes one layer of backslash escaping before wslpath sees the
    # argument. Keep the filesystem path untouched and escape only this argv.
    $escapedPathForWsl = $path.Replace('\', '\\')
    $result = Invoke-Gx1WslBounded -Arguments @('--', 'wslpath', '-u', $escapedPathForWsl) -TimeoutMilliseconds $WslTimeoutMilliseconds
    $linux = @($result.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
    if ($result.ExitCode -ne 0 -or $linux.Count -ne 1) {
        throw "Boot identity path conversion failed: $($result.StdErr.Trim())"
    }
    return [pscustomobject]@{ Windows = $path; Linux = $linux[0]; Payload = $value }
}
function Test-Gx1PrivateIpv4 {
    param([Parameter(Mandatory = $true)][string]$Address)
    $parsed = $null
    if (-not [Net.IPAddress]::TryParse($Address, [ref]$parsed) -or
        $parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) {
        return $false
    }
    $octets = $parsed.GetAddressBytes()
    return ($octets[0] -eq 10 -or
        ($octets[0] -eq 172 -and $octets[1] -ge 16 -and $octets[1] -le 31) -or
        ($octets[0] -eq 192 -and $octets[1] -eq 168))
}
function Assert-Gx1HostTelemetryBridgeV4 {
    param([Parameter(Mandatory = $true)][object]$Status)
    $bridgeRoot = Join-Path $env:ProgramData 'GX1\HostTelemetryBridgeV4'
    $configurationPath = Join-Path $bridgeRoot 'bridge-config.json'
    $expectedServicePath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeService.ps1'
    $expectedRunnerPath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeRunner.ps1'
    $expectedLogPath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeService.log'
    $expectedCertificateWindowsPath = Join-Path $bridgeRoot 'GX1HostTelemetryBridgePublic.pem'
    $expectedCertificateWslPath = '/mnt/c/ProgramData/GX1/HostTelemetryBridgeV4/GX1HostTelemetryBridgePublic.pem'
    $expectedQueryPath = $SourceRepo.TrimEnd('/') + '/scripts/gx1_host_telemetry_bridge_query.sh'
    $expectedListenAddress = '172.30.224.1'
    $expectedClientAddress = '172.30.231.75'
    $expectedLoopbackEndpoint = 'http://127.0.0.1:38127/gx1/v1/telemetry/'
    $expectedWslEndpoint = 'http://172.30.224.1:38128/gx1/v1/telemetry/'
    $expectedTaskName = 'GX1HostTelemetryBridge'

    foreach ($path in @($configurationPath, $expectedServicePath, $expectedRunnerPath, $expectedLogPath, $expectedCertificateWindowsPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
            ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'HostTelemetryBridgeV4 required file is missing or is a reparse point'
        }
    }
    $configuration = Get-Content -LiteralPath $configurationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $expectedConfigurationFields = @(
        'certificate_thumbprint', 'expected_gpu_name', 'gpu_index', 'loopback_endpoint',
        'runner_path', 'schema_version', 'service_log_path', 'service_path',
        'wsl_client_address', 'wsl_endpoint', 'wsl_listen_address', 'wsl_proxy_port',
        'wsl_transport'
    )
    $actualConfigurationFields = @($configuration.PSObject.Properties.Name | Sort-Object)
    if ((Compare-Object -ReferenceObject $expectedConfigurationFields -DifferenceObject $actualConfigurationFields).Count -ne 0 -or
        $configuration.schema_version -cne 'gx1_host_telemetry_bridge_install_v1' -or
        $configuration.expected_gpu_name -cne 'NVIDIA GeForce RTX 3090' -or
        [int]$configuration.gpu_index -ne 0 -or
        $configuration.loopback_endpoint -cne $expectedLoopbackEndpoint -or
        $configuration.wsl_endpoint -cne $expectedWslEndpoint -or
        $configuration.wsl_listen_address -cne $expectedListenAddress -or
        $configuration.wsl_client_address -cne $expectedClientAddress -or
        [int]$configuration.wsl_proxy_port -ne 38128 -or
        $configuration.wsl_transport -cne 'v4tov4_portproxy_to_windows_loopback' -or
        -not [StringComparer]::OrdinalIgnoreCase.Equals([string]$configuration.service_path, $expectedServicePath) -or
        -not [StringComparer]::OrdinalIgnoreCase.Equals([string]$configuration.runner_path, $expectedRunnerPath) -or
        -not [StringComparer]::OrdinalIgnoreCase.Equals([string]$configuration.service_log_path, $expectedLogPath) -or
        -not (Test-Gx1PrivateIpv4 -Address ([string]$configuration.wsl_listen_address)) -or
        -not (Test-Gx1PrivateIpv4 -Address ([string]$configuration.wsl_client_address))) {
        throw 'HostTelemetryBridgeV4 configuration differs from the exact campaign transport'
    }

    $queryBinding = $Status.signed_guard_sources.query
    $certificateBinding = $Status.signed_guard_sources.certificate
    if ($queryBinding.path -cne $expectedQueryPath -or
        $certificateBinding.path -cne $expectedCertificateWslPath -or
        $Status.gpu_uuid -notmatch '^GPU-[0-9a-fA-F-]{36}$' -or
        (Get-FileHash -LiteralPath $expectedCertificateWindowsPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$certificateBinding.sha256) {
        throw 'Host telemetry query, certificate, or GPU identity differs from the source-bound campaign'
    }
    $thumbprint = [string]$configuration.certificate_thumbprint
    if ($thumbprint -notmatch '^[0-9A-F]{40}$') { throw 'Host telemetry certificate thumbprint is malformed' }
    $signingCertificate = Get-Item -LiteralPath ("Cert:\LocalMachine\My\$thumbprint") -ErrorAction Stop
    if (-not $signingCertificate.HasPrivateKey) { throw 'Host telemetry signing certificate has no private key' }

    $task = Get-ScheduledTask -TaskName $expectedTaskName -ErrorAction Stop
    $expectedPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $expectedTaskArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$expectedRunnerPath`""
    if ($task.State -ne 'Running' -or
        $task.Principal.UserId -cne 'SYSTEM' -or
        [string]$task.Principal.LogonType -cne 'ServiceAccount' -or
        [string]$task.Principal.RunLevel -cne 'Highest' -or
        @($task.Actions).Count -ne 1 -or
        -not [StringComparer]::OrdinalIgnoreCase.Equals([string]$task.Actions[0].Execute, $expectedPowerShell) -or
        [string]$task.Actions[0].Arguments -cne $expectedTaskArguments -or
        @($task.Triggers).Count -ne 1 -or
        $task.Triggers[0].CimClass.CimClassName -cne 'MSFT_TaskBootTrigger' -or
        $task.Settings.StartWhenAvailable -ne $true -or
        [string]$task.Settings.MultipleInstances -cne 'IgnoreNew') {
        throw 'HostTelemetryBridgeV4 scheduled task assumptions differ'
    }
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort 38127 -ErrorAction Stop)
    if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -cne '127.0.0.1') {
        throw 'HostTelemetryBridgeV4 loopback listener is unavailable or widened'
    }

    $addressResult = Invoke-Gx1WslBounded -Arguments @('--', '/bin/hostname', '-I') -TimeoutMilliseconds 8000
    $wslAddresses = @($addressResult.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
    if ($addressResult.ExitCode -ne 0 -or $wslAddresses.Count -ne 1 -or
        -not (([string]$wslAddresses[0]).Split(' ', [StringSplitOptions]::RemoveEmptyEntries) -ccontains $expectedClientAddress)) {
        throw 'Current WSL client address differs from HostTelemetryBridgeV4 configuration'
    }
    $routeResult = Invoke-Gx1WslBounded -Arguments @('--', '/usr/sbin/ip', '-4', 'route', 'show', 'default') -TimeoutMilliseconds 8000
    $defaultRoutes = @($routeResult.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
    if ($routeResult.ExitCode -ne 0 -or $defaultRoutes.Count -ne 1 -or
        [string]$defaultRoutes[0] -notmatch '^default via ([0-9.]+) dev [A-Za-z0-9_.-]+(?: .*)?$' -or
        $Matches[1] -cne $expectedListenAddress) {
        throw 'Current WSL gateway differs from HostTelemetryBridgeV4 configuration'
    }

    $firewallRules = @(Get-NetFirewallRule -DisplayName 'GX1HostTelemetryBridge-Wsl-38128' -ErrorAction Stop)
    if ($firewallRules.Count -ne 1 -or [string]$firewallRules[0].Enabled -cne 'True' -or
        [string]$firewallRules[0].Direction -cne 'Inbound' -or
        [string]$firewallRules[0].Action -cne 'Allow' -or
        [string]$firewallRules[0].Profile -cne 'Any' -or
        [string]$firewallRules[0].EdgeTraversalPolicy -cne 'Block') {
        throw 'HostTelemetryBridgeV4 firewall rule assumptions differ'
    }
    $addressFilters = @($firewallRules[0] | Get-NetFirewallAddressFilter -ErrorAction Stop)
    $portFilters = @($firewallRules[0] | Get-NetFirewallPortFilter -ErrorAction Stop)
    if ($addressFilters.Count -ne 1 -or $portFilters.Count -ne 1 -or
        [string]$addressFilters[0].LocalAddress -cne $expectedListenAddress -or
        [string]$addressFilters[0].RemoteAddress -cne $expectedClientAddress -or
        [string]$portFilters[0].Protocol -cne 'TCP' -or
        [string]$portFilters[0].LocalPort -cne '38128') {
        throw 'HostTelemetryBridgeV4 firewall scope differs'
    }
    return [pscustomobject]@{
        Url = $expectedWslEndpoint
        ListenAddress = $expectedListenAddress
        ListenPort = 38128
        ConnectAddress = '127.0.0.1'
        ConnectPort = 38127
        QueryPath = [string]$queryBinding.path
        QuerySha256 = [string]$queryBinding.sha256
        CertificatePath = [string]$certificateBinding.path
        CertificateSha256 = [string]$certificateBinding.sha256
        GpuUuid = [string]$Status.gpu_uuid
    }
}
function Join-Gx1NativeArguments {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    foreach ($argument in $Arguments) {
        if ([string]::IsNullOrEmpty($argument) -or $argument -match '["\s]') {
            throw 'Native process argument is empty or requires forbidden quoting'
        }
    }
    return [string]::Join(' ', $Arguments)
}
function Invoke-Gx1NativeProcessBounded {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][int]$TimeoutMilliseconds
    )
    if ($TimeoutMilliseconds -le 0) { throw 'Native process timeout must be positive' }
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = Join-Gx1NativeArguments -Arguments $ArgumentList
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $process.Start()) { throw "Failed to start bounded process: $FilePath" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            try { $process.Kill() } catch {}
            throw "Bounded process timed out after $($TimeoutMilliseconds)ms: $FilePath"
        }
        $remaining = [int][Math]::Max(1, $TimeoutMilliseconds - $clock.ElapsedMilliseconds)
        $outputTasks = [Threading.Tasks.Task[]]@($stdoutTask, $stderrTask)
        if (-not [Threading.Tasks.Task]::WaitAll($outputTasks, $remaining)) {
            try { $process.Kill() } catch {}
            throw "Bounded process output drain timed out after $($TimeoutMilliseconds)ms: $FilePath"
        }
        return [pscustomobject]@{
            ExitCode = [int]$process.ExitCode
            StdOut = [string]$stdoutTask.Result
            StdErr = [string]$stderrTask.Result
        }
    } finally {
        $process.Dispose()
    }
}
function Invoke-Gx1WslBounded {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][int]$TimeoutMilliseconds
    )
    if ($Distro -notmatch '^[A-Za-z0-9_.-]+$' -or $LinuxUser -notmatch '^[A-Za-z0-9_.-]+$') {
        throw 'WSL distro or user is unsafe for direct process arguments'
    }
    $allArguments = @('-d', $Distro, '-u', $LinuxUser) + $Arguments
    return Invoke-Gx1NativeProcessBounded -FilePath (Join-Path $env:WINDIR 'System32\wsl.exe') -ArgumentList $allArguments -TimeoutMilliseconds $TimeoutMilliseconds
}
function Get-Gx1InitialCampaignState {
    # A BootTrigger may run before the WSL distro accepts its first command.
    # Retry both boot-path conversion and the initial read-only inspection under
    # one monotonic budget.  No ACTIVE state or CUDA process exists at this point.
    $deadlineMilliseconds = 60000
    $nativeCallLimitMilliseconds = 8000
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $lastFailure = 'cold WSL readiness was not observed'
    while ($clock.ElapsedMilliseconds -lt $deadlineMilliseconds) {
        try {
            $remaining = [int][Math]::Max(1, $deadlineMilliseconds - $clock.ElapsedMilliseconds)
            $callLimit = [int][Math]::Min($nativeCallLimitMilliseconds, $remaining)
            $boot = Write-Gx1BootIdentity -WslTimeoutMilliseconds $callLimit

            $remaining = [int][Math]::Max(1, $deadlineMilliseconds - $clock.ElapsedMilliseconds)
            $callLimit = [int][Math]::Min($nativeCallLimitMilliseconds, $remaining)
            $status = Invoke-Gx1Json -Arguments @(
                'inspect', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
                '--boot-json', $boot.Linux
            ) -TimeoutMilliseconds $callLimit
            return [pscustomobject]@{ Boot = $boot; Status = $status }
        } catch {
            $lastFailure = $_.Exception.Message
        }
        $remaining = [int]($deadlineMilliseconds - $clock.ElapsedMilliseconds)
        if ($remaining -gt 0) {
            Start-Sleep -Milliseconds ([int][Math]::Min(2000, $remaining))
        }
    }
    throw "Initial cold-WSL campaign inspection failed after bounded retry: $lastFailure"
}
function Wait-Gx1HostTelemetryBridgeV4BootReady {
    # The telemetry service and campaign controller are independent boot tasks.
    # Wait only for their dynamic boot state here; Assert below revalidates the
    # complete immutable task, transport, source, certificate, and firewall shape.
    $expectedTaskName = 'GX1HostTelemetryBridge'
    $expectedClientAddress = '172.30.231.75'
    $expectedListenAddress = '172.30.224.1'
    $deadlineMilliseconds = 60000
    $nativeCallLimitMilliseconds = 8000
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $lastFailure = 'boot readiness was not observed'
    while ($clock.ElapsedMilliseconds -lt $deadlineMilliseconds) {
        try {
            $task = Get-ScheduledTask -TaskName $expectedTaskName -ErrorAction Stop
            if ($task.State -ne 'Running') {
                throw "telemetry task state is $($task.State)"
            }
            $listeners = @(Get-NetTCPConnection -State Listen -LocalPort 38127 -ErrorAction Stop)
            if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -cne '127.0.0.1') {
                throw 'loopback listener is not ready'
            }
            $remaining = [int][Math]::Max(1, $deadlineMilliseconds - $clock.ElapsedMilliseconds)
            $callLimit = [int][Math]::Min($nativeCallLimitMilliseconds, $remaining)
            $addressResult = Invoke-Gx1WslBounded -Arguments @('--', '/bin/hostname', '-I') -TimeoutMilliseconds $callLimit
            $wslAddresses = @($addressResult.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
            if ($addressResult.ExitCode -ne 0 -or $wslAddresses.Count -ne 1 -or
                -not (([string]$wslAddresses[0]).Split(' ', [StringSplitOptions]::RemoveEmptyEntries) -ccontains $expectedClientAddress)) {
                throw "WSL client address is not ready: $($addressResult.StdErr.Trim())"
            }
            $remaining = [int][Math]::Max(1, $deadlineMilliseconds - $clock.ElapsedMilliseconds)
            $callLimit = [int][Math]::Min($nativeCallLimitMilliseconds, $remaining)
            $routeResult = Invoke-Gx1WslBounded -Arguments @('--', '/usr/sbin/ip', '-4', 'route', 'show', 'default') -TimeoutMilliseconds $callLimit
            $defaultRoutes = @($routeResult.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
            if ($routeResult.ExitCode -ne 0 -or $defaultRoutes.Count -ne 1 -or
                [string]$defaultRoutes[0] -notmatch '^default via ([0-9.]+) dev [A-Za-z0-9_.-]+(?: .*)?$' -or
                $Matches[1] -cne $expectedListenAddress) {
                throw "WSL gateway is not ready: $($routeResult.StdErr.Trim())"
            }
            return
        } catch {
            $lastFailure = $_.Exception.Message
        }
        $remaining = [int]($deadlineMilliseconds - $clock.ElapsedMilliseconds)
        if ($remaining -gt 0) {
            Start-Sleep -Milliseconds ([int][Math]::Min(2000, $remaining))
        }
    }
    throw "HostTelemetryBridgeV4 boot readiness failed after bounded retry: $lastFailure"
}
function Reset-Gx1HostTelemetryPortProxy {
    param([Parameter(Mandatory = $true)][object]$Bridge)
    $netsh = Join-Path $env:WINDIR 'System32\netsh.exe'
    # Only the exact private V4 rule is touched. Missing-rule deletion is benign;
    # the checked add and signed end-to-end probe below remain fail closed.
    & $netsh interface portproxy delete v4tov4 "listenaddress=$($Bridge.ListenAddress)" "listenport=$($Bridge.ListenPort)" protocol=tcp 2>$null | Out-Null
    $added = @(& $netsh interface portproxy add v4tov4 `
        "listenaddress=$($Bridge.ListenAddress)" `
        "listenport=$($Bridge.ListenPort)" `
        "connectaddress=$($Bridge.ConnectAddress)" `
        "connectport=$($Bridge.ConnectPort)" `
        protocol=tcp 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Exact HostTelemetryBridgeV4 portproxy refresh failed: $(($added | Out-String).Trim())"
    }
}
function Confirm-Gx1SignedHostTelemetryReady {
    param([Parameter(Mandatory = $true)][object]$Status)
    Wait-Gx1HostTelemetryBridgeV4BootReady
    $bridge = Assert-Gx1HostTelemetryBridgeV4 -Status $Status
    Reset-Gx1HostTelemetryPortProxy -Bridge $bridge

    $queryHashResult = Invoke-Gx1WslBounded -Arguments @('--', '/usr/bin/sha256sum', $bridge.QueryPath) -TimeoutMilliseconds 8000
    $queryHashOutput = @($queryHashResult.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
    if ($queryHashResult.ExitCode -ne 0 -or $queryHashOutput.Count -ne 1 -or
        [string]$queryHashOutput[0] -notmatch '^([0-9a-f]{64})  ' -or
        $Matches[1] -cne $bridge.QuerySha256) {
        throw 'Canonical host telemetry query no longer matches the inspected source binding'
    }
    foreach ($attempt in 1..12) {
        $probeResult = Invoke-Gx1WslBounded -Arguments @(
            '--', $bridge.QueryPath, $bridge.Url, $bridge.CertificatePath,
            $bridge.CertificateSha256, $bridge.GpuUuid, '2'
        ) -TimeoutMilliseconds 8000
        $telemetry = @($probeResult.StdOut -split '\r?\n' | Where-Object { $_ -cne '' })
        $probeExitCode = $probeResult.ExitCode
        if ($probeExitCode -eq 0 -and $telemetry.Count -eq 1 -and
            [string]$telemetry[0] -match '^[0-9]+(?:\.[0-9]+)?,[0-9]+(?:\.[0-9]+)?,[0-9]+(?:\.[0-9]+)?,[0-9]+(?:\.[0-9]+)?,[0-9]+$') {
            return
        }
        if ($attempt -lt 12) { Start-Sleep -Seconds 2 }
    }
    throw 'Canonical signed HostTelemetryBridgeV4 readiness probe failed after bounded retry'
}
function Request-Gx1PhysicalReboot {
    param([object]$Boot)
    $prepared = Invoke-Gx1Json -Arguments @(
        'prepare-reboot', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
        '--boot-json', $Boot.Linux
    )
    & shutdown.exe /r /t 60 /d p:0:0 /c 'GX1 random-access campaign requires a fresh physical Windows boot'
    $shutdownExit = $LASTEXITCODE
    if ($shutdownExit -ne 0) { throw 'shutdown.exe rejected physical reboot request' }
    [void](Invoke-Gx1Json -Arguments @(
        'confirm-reboot', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
        '--request-nonce', [string]$prepared.intent.request_nonce,
        '--shutdown-exit-code', [string]$shutdownExit
    ))
}
$initial = Get-Gx1InitialCampaignState
$boot = $initial.Boot
$status = $initial.Status
$controllerSourceRoot = [IO.Path]::GetFullPath($WindowsControllerSourceRoot).TrimEnd('\')
if ($controllerSourceRoot -notmatch '^[Cc]:\\' -or
    -not (Test-Path -LiteralPath $controllerSourceRoot -PathType Container) -or
    ((Get-Item -LiteralPath $controllerSourceRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Windows controller source root must be an existing local C: directory without a reparse point'
}
$controllerBinding = $status.controller_sources.controller
$observerBinding = $status.controller_sources.observer
$expectedControllerSource = Join-Path $controllerSourceRoot 'scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1'
$observerSource = Join-Path $controllerSourceRoot 'scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1'
foreach ($source in @($expectedControllerSource, $observerSource)) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or
        ((Get-Item -LiteralPath $source -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Windows controller staging source is unavailable or is a reparse point'
    }
}
if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
        [IO.Path]::GetFullPath($PSCommandPath),
        [IO.Path]::GetFullPath($expectedControllerSource))) {
    throw 'Running campaign controller is outside the explicit staging root'
}
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$controllerBinding.sha256 -or
    (Get-FileHash -LiteralPath $observerSource -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$observerBinding.sha256) {
    throw 'Campaign controller or observer differs from immutable plan binding'
}
if ($status.policy.physical_power_limit_w -ne 160 -or
    $status.policy.maximum_actual_power_draw_w -ne 170 -or
    $status.policy.signed_local_telemetry_seconds -ne 1 -or
    $status.policy.human_status_seconds -ne 900) {
    throw 'Campaign safety policy differs'
}
if ($status.action.decision -ceq 'REBOOT_REQUIRED') {
    Request-Gx1PhysicalReboot -Boot $boot
    exit 0
}
if ($status.action.decision -ceq 'COMPLETE' -or $status.action.decision -like 'BLOCKED*') {
    $status | ConvertTo-Json -Depth 16 -Compress
    exit 0
}
if ($status.action.decision -cne 'LAUNCH') { throw 'Campaign returned no admissible action' }
# The readiness probe must precede begin: begin publishes ACTIVE_INVOCATION.json.
# Any bridge, proxy, source, certificate, or signed-query failure therefore
# leaves the campaign non-active and CUDA is never invoked.
Confirm-Gx1SignedHostTelemetryReady -Status $status
$begin = Invoke-Gx1Json -Arguments @(
    'begin', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--boot-json', $boot.Linux
)
$invocation = $begin.invocation
$argv = @($invocation.launcher_argv | ForEach-Object { [string]$_ })
$progressWindows = Convert-Gx1WslPath -LinuxPath ([string]$invocation.progress_path)
$guardWindows = Convert-Gx1WslPath -LinuxPath ([string]$invocation.guard_log_path)
$runtimeWindows = Split-Path -Parent $progressWindows
$statusJson = Join-Path $runtimeWindows ('STATUS-{0}.json' -f $invocation.invocation_id)
$humanJsonl = Join-Path $runtimeWindows 'HUMAN_STATUS.jsonl'
if (Test-Path -LiteralPath $guardWindows) { throw 'Guard log already exists before invocation' }
$trainerArguments = @('-d', $Distro, '-u', $LinuxUser, '--cd', $SourceRepo, '--') + $argv
$priorPlanSha = $env:GX1_CAMPAIGN_PLAN_SHA256
$priorPlanPath = $env:GX1_CAMPAIGN_PLAN_PATH
$priorPlanFileSha = $env:GX1_CAMPAIGN_PLAN_FILE_SHA256
$priorInvocationSha = $env:GX1_CAMPAIGN_INVOCATION_SHA256
$priorGuardLogPath = $env:GX1_CAMPAIGN_GUARD_LOG_PATH
$priorWslEnv = $env:WSLENV
$campaignWslEnv = 'GX1_CAMPAIGN_PLAN_SHA256:GX1_CAMPAIGN_PLAN_PATH:GX1_CAMPAIGN_PLAN_FILE_SHA256:GX1_CAMPAIGN_INVOCATION_SHA256:GX1_CAMPAIGN_GUARD_LOG_PATH'
try {
    $env:GX1_CAMPAIGN_PLAN_SHA256 = [string]$status.plan_sha256
    if (-not $PlanJson.StartsWith('/')) { throw 'Campaign plan must be an absolute WSL path' }
    $env:GX1_CAMPAIGN_PLAN_PATH = $PlanJson
    $env:GX1_CAMPAIGN_PLAN_FILE_SHA256 = $PlanFileSha256
    $env:GX1_CAMPAIGN_INVOCATION_SHA256 = [string]$invocation.invocation_sha256
    $env:GX1_CAMPAIGN_GUARD_LOG_PATH = [string]$invocation.guard_log_path
    $env:WSLENV = if ([string]::IsNullOrWhiteSpace($priorWslEnv)) {
        $campaignWslEnv
    }
    else {
        "$priorWslEnv`:$campaignWslEnv"
    }
    $trainer = Start-Process -FilePath 'wsl.exe' -ArgumentList $trainerArguments -PassThru -NoNewWindow
}
finally {
    $env:GX1_CAMPAIGN_PLAN_SHA256 = $priorPlanSha
    $env:GX1_CAMPAIGN_PLAN_PATH = $priorPlanPath
    $env:GX1_CAMPAIGN_PLAN_FILE_SHA256 = $priorPlanFileSha
    $env:GX1_CAMPAIGN_INVOCATION_SHA256 = $priorInvocationSha
    $env:GX1_CAMPAIGN_GUARD_LOG_PATH = $priorGuardLogPath
    $env:WSLENV = $priorWslEnv
}
$observerScript = Join-Path $controllerSourceRoot 'scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1'
$observer = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $observerScript,
    '-TrainerProcessId', [string]$trainer.Id,
    '-PlanSha256', [string]$status.plan_sha256,
    '-InvocationSha256', [string]$invocation.invocation_sha256,
    '-ProgressJson', $progressWindows,
    '-StatusJson', $statusJson,
    '-HumanStatusJsonl', $humanJsonl
) -PassThru -NoNewWindow
$trainer.WaitForExit()
$observer.WaitForExit()
$outcome = if ($trainer.ExitCode -eq 0 -and $observer.ExitCode -eq 0) {
    [string]$invocation.expected_success_outcome
}
else {
    'FAILED'
}
$recorded = Invoke-Gx1Json -Arguments @(
    'record', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--trainer-guard-exit-code', [string]$trainer.ExitCode,
    '--progress-observer-exit-code', [string]$observer.ExitCode,
    '--outcome', $outcome
)
if ($outcome -eq 'FAILED') {
    $recorded | ConvertTo-Json -Depth 16 -Compress
    exit 2
}
$after = Invoke-Gx1Json -Arguments @(
    'inspect', '--plan-json', $PlanJson, '--plan-file-sha256', $PlanFileSha256,
    '--boot-json', $boot.Linux
)
if ($after.action.decision -eq 'COMPLETE') {
    $recorded | ConvertTo-Json -Depth 16 -Compress
    exit 0
}
Request-Gx1PhysicalReboot -Boot $boot
$recorded | ConvertTo-Json -Depth 16 -Compress
