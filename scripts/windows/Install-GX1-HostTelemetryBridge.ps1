<#
.SYNOPSIS
  Installs the least-privilege Windows host signer for GX1 GPU telemetry.

.DESCRIPTION
  This elevated native-Windows script creates a non-exportable RSA signing
  certificate in LocalMachine, stores a SYSTEM-run loopback service under
  C:\ProgramData\GX1\HostTelemetryBridge, and exports only its public
  certificate.  The service loads the already hash-verified LibreHardwareMonitor
  library solely to obtain Nvidia "GPU Memory Junction"; the remaining physical
  fields come from native Windows nvidia-smi.

  It is not a training command and does not change the GPU power limit.  The
  Linux canonical guard will remain locked until the exported public-certificate
  SHA-256 is source-bound in a later commit and a signed bridge probe succeeds.
#>

[CmdletBinding()]
param(
    [string]$ExpectedGpuName = 'NVIDIA GeForce RTX 3090',
    [ValidateRange(0, 31)]
    [int]$GpuIndex = 0,
    [ValidateRange(1024, 65535)]
    [int]$Port = 38127,
    [switch]$RotateCertificate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script from native Windows PowerShell as Administrator. No bridge was installed.'
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    $output = & $FilePath @ArgumentList 2>&1
    if ($LASTEXITCODE -ne 0) {
        $rendered = ($output | Out-String).Trim()
        throw "Native command failed ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')`n$rendered"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

function Get-BridgeCertificate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath,
        [switch]$ForceRotate
    )

    if ((Test-Path -LiteralPath $ConfigPath -PathType Leaf) -and -not $ForceRotate) {
        $prior = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
        $thumbprint = [string]$prior.certificate_thumbprint
        if ($thumbprint -notmatch '^[A-Fa-f0-9]{40}$') {
            throw 'Existing bridge configuration contains an invalid certificate thumbprint.'
        }
        $existing = Get-Item -LiteralPath "Cert:\LocalMachine\My\$thumbprint" -ErrorAction SilentlyContinue
        if ($null -eq $existing -or -not $existing.HasPrivateKey) {
            throw 'Existing bridge certificate is unavailable or has no private key. Rerun explicitly with -RotateCertificate.'
        }
        return $existing
    }

    $certificateParameters = @{
        Type = 'Custom'
        Subject = 'CN=GX1 Host Telemetry Bridge'
        CertStoreLocation = 'Cert:\LocalMachine\My'
        KeyAlgorithm = 'RSA'
        KeyLength = 3072
        HashAlgorithm = 'SHA256'
        KeyUsage = 'DigitalSignature'
        KeyExportPolicy = 'NonExportable'
        KeySpec = 'Signature'
        NotAfter = (Get-Date).AddYears(3)
    }
    return New-SelfSignedCertificate @certificateParameters
}

function Export-PublicCertificatePem {
    param(
        [Parameter(Mandatory = $true)]
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    $der = $Certificate.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
    $base64 = [Convert]::ToBase64String(
        $der,
        [System.Base64FormattingOptions]::InsertLineBreaks
    )
    $pem = "-----BEGIN CERTIFICATE-----`n$base64`n-----END CERTIFICATE-----`n"
    [System.IO.File]::WriteAllText(
        $DestinationPath,
        $pem,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Set-BridgeDirectoryAcl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BridgeRoot
    )

    # `/T` is intentional: every persisted service/config/public-key file gets
    # the same restrictive ACL, rather than merely inheriting a new directory
    # rule.  Users retain read/execute solely so the Linux verifier can read
    # the public certificate; only SYSTEM/Administrators can change anything.
    $icacls = Join-Path $env:WINDIR 'System32\icacls.exe'
    # Use well-known SID syntax (`*SID`) rather than English local-group names:
    # on this host the Administrator group is localized to Norwegian.
    Invoke-NativeChecked -FilePath $icacls -ArgumentList @(
        $BridgeRoot,
        '/inheritance:r',
        '/grant:r', '*S-1-5-18:(OI)(CI)F',
        '*S-1-5-32-544:(OI)(CI)F',
        '*S-1-5-32-545:(OI)(CI)RX',
        '/T', '/C'
    ) | Out-Null
}

function Repair-ExistingBridgeDirectoryAccess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BridgeRoot
    )

    # A prior interrupted install can leave this narrowly scoped directory
    # protected before its configuration has been written successfully.  An
    # elevated administrator reclaims ownership of this exact GX1 directory,
    # then reapplies the least-privilege ACL before reading any prior state.
    # This deliberately does not touch any other ProgramData content.
    if (-not (Test-Path -LiteralPath $BridgeRoot -PathType Container)) {
        return
    }
    $takeown = Join-Path $env:WINDIR 'System32\takeown.exe'
    Invoke-NativeChecked -FilePath $takeown -ArgumentList @(
        '/F', $BridgeRoot,
        '/A', '/R', '/D', 'Y'
    ) | Out-Null
    Set-BridgeDirectoryAcl -BridgeRoot $BridgeRoot
}

Assert-Administrator

$lhmRoot = Join-Path $env:ProgramData 'GX1\LibreHardwareMonitor\v0.9.6'
$lhmLibrary = Join-Path $lhmRoot 'LibreHardwareMonitorLib.dll'
if (-not (Test-Path -LiteralPath $lhmLibrary -PathType Leaf)) {
    throw "The verified LibreHardwareMonitor library is unavailable at $lhmLibrary. Run Install-GX1-HostTelemetry.ps1 first."
}
$nativeSmi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
if (-not (Test-Path -LiteralPath $nativeSmi -PathType Leaf)) {
    throw "Native nvidia-smi.exe is unavailable at $nativeSmi"
}

$bridgeRoot = Join-Path $env:ProgramData 'GX1\HostTelemetryBridge'
New-Item -ItemType Directory -Path $bridgeRoot -Force | Out-Null
Repair-ExistingBridgeDirectoryAccess -BridgeRoot $bridgeRoot
$configPath = Join-Path $bridgeRoot 'bridge-config.json'
$servicePath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeService.ps1'
$runnerPath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeRunner.ps1'
$serviceLogPath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeService.log'
$publicCertificatePath = Join-Path $bridgeRoot 'GX1HostTelemetryBridgePublic.pem'
$endpoint = "http://127.0.0.1:$Port/gx1/v1/telemetry/"
$certificate = Get-BridgeCertificate -ConfigPath $configPath -ForceRotate:$RotateCertificate

$serviceSource = @'
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CertificateThumbprint,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedGpuName,
    [ValidateRange(0, 31)]
    [int]$GpuIndex,
    [ValidateRange(1024, 65535)]
    [int]$Port
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$responseSchema = 'gx1_host_gpu_telemetry_v1'
$requestSchema = 'gx1_host_gpu_telemetry_request_v1'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Add-BridgeUpdateVisitor {
    param([string]$LibraryPath)
    Add-Type -Path $LibraryPath
    if ($null -eq ('Gx1HostTelemetryUpdateVisitor' -as [type])) {
        Add-Type -TypeDefinition @"
using LibreHardwareMonitor.Hardware;

public sealed class Gx1HostTelemetryUpdateVisitor : IVisitor
{
    public void VisitComputer(IComputer computer) { computer.Traverse(this); }
    public void VisitHardware(IHardware hardware)
    {
        hardware.Update();
        foreach (IHardware subHardware in hardware.SubHardware)
            subHardware.Accept(this);
    }
    public void VisitSensor(ISensor sensor) { }
    public void VisitParameter(IParameter parameter) { }
}
"@ -ReferencedAssemblies $LibraryPath
    }
}

function Write-BridgeResponse {
    param(
        [Parameter(Mandatory = $true)]
        [System.Net.HttpListenerContext]$Context,
        [Parameter(Mandatory = $true)]
        [int]$StatusCode,
        [Parameter(Mandatory = $true)]
        [string]$Body
    )

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
    $Context.Response.StatusCode = $StatusCode
    $Context.Response.ContentType = 'application/json; charset=utf-8'
    $Context.Response.ContentLength64 = $bytes.Length
    $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $Context.Response.Close()
}

function Require-RequestNonce {
    param([System.Net.HttpListenerRequest]$Request)
    if ($Request.HttpMethod -ne 'POST' -or $Request.ContentLength64 -lt 1 -or $Request.ContentLength64 -gt 512) {
        throw 'invalid request envelope'
    }
    $reader = [System.IO.StreamReader]::new($Request.InputStream, [System.Text.Encoding]::UTF8, $false, 512, $true)
    try {
        $body = $reader.ReadToEnd()
    }
    finally {
        $reader.Dispose()
    }
    $parsed = $body | ConvertFrom-Json -ErrorAction Stop
    $names = @($parsed.PSObject.Properties.Name | Sort-Object)
    if (($names -join ',') -ne 'request_nonce,schema_version' -or
        [string]$parsed.schema_version -ne $requestSchema -or
        [string]$parsed.request_nonce -notmatch '^[0-9a-f]{64}$') {
        throw 'invalid request schema'
    }
    return [string]$parsed.request_nonce
}

function Get-NativeGpuTelemetry {
    $nativeSmi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
    $rows = @(& $nativeSmi -i "$GpuIndex" --query-gpu=name,uuid,temperature.gpu,power.draw,power.limit,memory.used --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -ne 1) {
        throw 'native GPU telemetry is unavailable'
    }
    $fields = @($rows[0].ToString().Split(',') | ForEach-Object { $_.Trim() })
    if ($fields.Count -ne 6 -or $fields[0] -ne $ExpectedGpuName -or $fields[1] -notmatch '^GPU-[0-9a-fA-F-]{36}$') {
        throw 'native GPU identity did not match'
    }
    foreach ($index in 2..5) {
        if ($fields[$index] -notmatch '^[0-9]+([.][0-9]+)?$') {
            throw 'native GPU telemetry contains a non-numeric value'
        }
    }
    $core = [double]::Parse($fields[2], [System.Globalization.NumberStyles]::Float, $invariant)
    $draw = [double]::Parse($fields[3], [System.Globalization.NumberStyles]::Float, $invariant)
    $limit = [double]::Parse($fields[4], [System.Globalization.NumberStyles]::Float, $invariant)
    $used = [int64]::Parse($fields[5], [System.Globalization.NumberStyles]::Integer, $invariant)
    if ($core -lt 0 -or $draw -le 0 -or $limit -le 0 -or $used -lt 0) {
        throw 'native GPU telemetry violates finite physical bounds'
    }
    return [pscustomobject]@{
        uuid = $fields[1]
        core_temp_c = $core
        power_draw_w = $draw
        power_limit_w = $limit
        memory_used_mib = $used
    }
}

function Get-MemoryJunctionTemperature {
    param([LibreHardwareMonitor.Hardware.Computer]$Computer)
    $Computer.Accept([Gx1HostTelemetryUpdateVisitor]::new())
    $gpus = @($Computer.Hardware | Where-Object {
        $_.HardwareType.ToString() -eq 'GpuNvidia' -and $_.Name -eq $ExpectedGpuName
    })
    if ($gpus.Count -ne 1) {
        throw 'LibreHardwareMonitor GPU identity did not match'
    }
    $junction = @($gpus[0].Sensors | Where-Object {
        $_.SensorType.ToString() -eq 'Temperature' -and $_.Name -eq 'GPU Memory Junction'
    })
    if ($junction.Count -ne 1 -or $null -eq $junction[0].Value) {
        throw 'LibreHardwareMonitor memory junction is unavailable'
    }
    $value = [double]$junction[0].Value
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0) {
        throw 'LibreHardwareMonitor memory junction is non-finite'
    }
    return $value
}

function Format-CanonicalFloat {
    param([double]$Value)
    return $Value.ToString('F6', $invariant)
}

function Sign-BridgePayload {
    param(
        [System.Security.Cryptography.RSA]$Rsa,
        [string]$Payload
    )
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Payload)
    $signature = $Rsa.SignData(
        $bytes,
        [System.Security.Cryptography.HashAlgorithmName]::SHA256,
        [System.Security.Cryptography.RSASignaturePadding]::Pkcs1
    )
    return [Convert]::ToBase64String($signature)
}

if ($CertificateThumbprint -notmatch '^[A-Fa-f0-9]{40}$') {
    throw 'certificate thumbprint is invalid'
}
$certificate = Get-Item -LiteralPath "Cert:\LocalMachine\My\$CertificateThumbprint" -ErrorAction Stop
$rsa = [System.Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($certificate)
if ($null -eq $rsa) {
    throw 'bridge certificate has no RSA private key'
}
$lhmRoot = Join-Path $env:ProgramData 'GX1\LibreHardwareMonitor\v0.9.6'
$lhmLibrary = Join-Path $lhmRoot 'LibreHardwareMonitorLib.dll'
if (-not (Test-Path -LiteralPath $lhmLibrary -PathType Leaf)) {
    throw 'LibreHardwareMonitor library is unavailable'
}
Add-BridgeUpdateVisitor -LibraryPath $lhmLibrary
$computer = [LibreHardwareMonitor.Hardware.Computer]::new()
$computer.IsGpuEnabled = $true
$computer.Open()
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("http://127.0.0.1:$Port/gx1/v1/telemetry/")

try {
    $listener.Start()
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        try {
            $nonce = Require-RequestNonce -Request $context.Request
            $native = Get-NativeGpuTelemetry
            $memory = Get-MemoryJunctionTemperature -Computer $computer
            $observed = [int64][Math]::Floor(
                ([System.Diagnostics.Stopwatch]::GetTimestamp() * 1000.0) / [System.Diagnostics.Stopwatch]::Frequency
            )
            $coreField = Format-CanonicalFloat -Value $native.core_temp_c
            $memoryField = Format-CanonicalFloat -Value $memory
            $drawField = Format-CanonicalFloat -Value $native.power_draw_w
            $limitField = Format-CanonicalFloat -Value $native.power_limit_w
            $payload = @(
                $responseSchema,
                $nonce,
                $native.uuid,
                $coreField,
                $memoryField,
                $drawField,
                $limitField,
                [string]($native.memory_used_mib),
                [string]$observed
            ) -join "`n"
            $payload += "`n"
            $signature = Sign-BridgePayload -Rsa $rsa -Payload $payload
            $json = '{"schema_version":"' + $responseSchema + '","request_nonce":"' + $nonce + '","gpu_uuid":"' + $native.uuid + '","core_temp_c":' + $coreField + ',"memory_temp_c":' + $memoryField + ',"power_draw_w":' + $drawField + ',"power_limit_w":' + $limitField + ',"memory_used_mib":' + $native.memory_used_mib + ',"observed_monotonic_ms":' + $observed + ',"signature":"' + $signature + '"}'
            Write-BridgeResponse -Context $context -StatusCode 200 -Body $json
        }
        catch {
            Write-BridgeResponse -Context $context -StatusCode 503 -Body '{"error":"telemetry_unavailable"}'
        }
    }
}
finally {
    if ($listener.IsListening) {
        $listener.Stop()
    }
    $listener.Close()
    $computer.Close()
    $rsa.Dispose()
}
'@

[System.IO.File]::WriteAllText(
    $servicePath,
    $serviceSource,
    [System.Text.UTF8Encoding]::new($false)
)
$runnerSource = @"
`$ErrorActionPreference = 'Stop'
& '$servicePath' -CertificateThumbprint '$($certificate.Thumbprint)' -ExpectedGpuName '$ExpectedGpuName' -GpuIndex $GpuIndex -Port $Port *>> '$serviceLogPath'
exit `$LASTEXITCODE
"@
[System.IO.File]::WriteAllText(
    $runnerPath,
    $runnerSource,
    [System.Text.UTF8Encoding]::new($false)
)
[System.IO.File]::WriteAllText(
    $serviceLogPath,
    '',
    [System.Text.UTF8Encoding]::new($false)
)

$configuration = [ordered]@{
    schema_version = 'gx1_host_telemetry_bridge_install_v1'
    certificate_thumbprint = $certificate.Thumbprint
    expected_gpu_name = $ExpectedGpuName
    gpu_index = $GpuIndex
    endpoint = $endpoint
    service_path = $servicePath
    runner_path = $runnerPath
    service_log_path = $serviceLogPath
}
$configuration | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $configPath -Encoding UTF8
Export-PublicCertificatePem -Certificate $certificate -DestinationPath $publicCertificatePath
Set-BridgeDirectoryAcl -BridgeRoot $bridgeRoot

$netsh = Join-Path $env:WINDIR 'System32\netsh.exe'
# A first install has no URL ACL to remove; an old reservation is replaced in
# either case.  The subsequent add is the checked operation that matters.
& $netsh http delete urlacl "url=$endpoint" 2>$null | Out-Null
Invoke-NativeChecked -FilePath $netsh -ArgumentList @('http', 'add', 'urlacl', "url=$endpoint", 'user=NT AUTHORITY\SYSTEM') | Out-Null

$taskName = 'GX1HostTelemetryBridge'
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -ne $existingTask) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}
$powerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runnerPath`""
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 2
$task = Get-ScheduledTask -TaskName $taskName
if ($task.State -ne 'Running') {
    $diagnostic = ''
    if (Test-Path -LiteralPath $serviceLogPath -PathType Leaf) {
        $diagnostic = (Get-Content -LiteralPath $serviceLogPath -Tail 80 | Out-String).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($diagnostic)) {
        $diagnostic = 'No service output was captured; inspect task result with Get-ScheduledTaskInfo -TaskName GX1HostTelemetryBridge.'
    }
    throw "Host telemetry bridge task did not enter Running state (state=$($task.State)). Diagnostic output:`n$diagnostic"
}

$certificateSha256 = (Get-FileHash -LiteralPath $publicCertificatePath -Algorithm SHA256).Hash.ToLowerInvariant()
$report = [ordered]@{
    schema_version = 'gx1_host_telemetry_bridge_install_v1'
    task_name = $taskName
    endpoint = $endpoint
    public_certificate_windows_path = $publicCertificatePath
    public_certificate_wsl_path = '/mnt/c/ProgramData/GX1/HostTelemetryBridge/GX1HostTelemetryBridgePublic.pem'
    public_certificate_sha256 = $certificateSha256
    gpu_uuid_expected_from_sensor_bootstrap = 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
    next_gate = 'Linux signed bridge probe and source binding; this does not authorize canonical training.'
}
Write-Host ''
Write-Host 'GX1 host telemetry bridge installed:' -ForegroundColor Green
$report | ConvertTo-Json -Depth 3
