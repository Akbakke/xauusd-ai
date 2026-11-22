<#
.SYNOPSIS
  Installs the least-privilege Windows host signer for GX1 GPU telemetry.

.DESCRIPTION
  This elevated native-Windows script creates a non-exportable RSA signing
  certificate in LocalMachine, stores a SYSTEM-run loopback service under
  C:\ProgramData\GX1\HostTelemetryBridge, and exports only its public
  certificate.  The optional WslClientAddress mode adds an exact-address IPv4
  port proxy from the Windows WSL gateway to the existing loopback listener,
  plus an inbound firewall rule limited to that single WSL client address; it
  never listens on a LAN or wildcard address.  The service loads the already
  hash-verified LibreHardwareMonitor library for Nvidia "GPU Memory Junction"
  and CPU core/package temperature and package power. Native Windows memory
  counters and conservative maintenance facts share the same signed response.

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
    [ValidateRange(1024, 65535)]
    [int]$WslPort = 38128,
    [ValidatePattern('^[A-Za-z][A-Za-z0-9_-]{0,63}$')]
    [string]$BridgeDirectoryName = 'HostTelemetryBridge',
    [string]$WslClientAddress = '',
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

function Stop-ExistingBridgeTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskName
    )

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $task -or $task.State -ne 'Running') {
        return
    }
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    foreach ($attempt in 1..25) {
        Start-Sleep -Milliseconds 200
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($null -eq $task -or $task.State -ne 'Running') {
            return
        }
    }
    throw "Existing bridge task $TaskName did not stop within five seconds; refusing to replace files it may still own."
}

function Wait-ForExclusiveFileAccess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    foreach ($attempt in 1..25) {
        try {
            $stream = [System.IO.File]::Open(
                $Path,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
            $stream.Dispose()
            return
        }
        catch [System.IO.IOException] {
            Start-Sleep -Milliseconds 200
        }
    }
    throw "Bridge file remained locked after the old task stopped: $Path"
}

function ConvertTo-CanonicalIpv4 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    try {
        $address = [System.Net.IPAddress]::Parse($Value)
    }
    catch {
        throw "$Description is not a valid IPv4 address."
    }
    if ($address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
        throw "$Description must be an IPv4 address."
    }
    if ($address.Equals([System.Net.IPAddress]::Any) -or
        $address.Equals([System.Net.IPAddress]::Loopback) -or
        $address.GetAddressBytes()[0] -ge 224) {
        throw "$Description must be a specific unicast IPv4 address."
    }
    return $address.IPAddressToString
}

function Test-Ipv4SameSubnet {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FirstAddress,
        [Parameter(Mandatory = $true)]
        [string]$SecondAddress,
        [ValidateRange(1, 30)]
        [int]$PrefixLength
    )

    $first = [System.Net.IPAddress]::Parse($FirstAddress).GetAddressBytes()
    $second = [System.Net.IPAddress]::Parse($SecondAddress).GetAddressBytes()
    $bitsRemaining = $PrefixLength
    foreach ($index in 0..3) {
        $mask = if ($bitsRemaining -ge 8) {
            255
        }
        elseif ($bitsRemaining -le 0) {
            0
        }
        else {
            [byte]((0xff -shl (8 - $bitsRemaining)) -band 0xff)
        }
        if (($first[$index] -band $mask) -ne ($second[$index] -band $mask)) {
            return $false
        }
        $bitsRemaining -= 8
    }
    return $true
}

function Get-WslGatewayIpv4 {
    $candidates = @(
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | Where-Object {
            $_.InterfaceAlias -like 'vEthernet (WSL*' -and
            $_.IPAddress -notlike '169.254.*' -and
            $_.PrefixLength -ge 1 -and $_.PrefixLength -le 30
        }
    )
    if ($candidates.Count -ne 1) {
        throw "Could not identify exactly one IPv4 address on the Windows WSL gateway (found $($candidates.Count)). Do not expose the bridge manually."
    }
    return $candidates[0]
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

    # Use the Windows ACL API instead of an argument-sensitive `icacls` grant
    # sequence. A historical invocation could leave an empty DACL after
    # disabling inheritance, which denies every reader including the elevated
    # installer. The new directory receives its inheritable rules before any
    # service/config/public-key file is written.
    $installerSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    if ($installerSid -notmatch '^S-1-5-21-(?:[0-9]+-){3}[0-9]+$') {
        throw 'Unable to determine a valid local installer SID for bridge ACLs.'
    }
    $inherit = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
        [Security.AccessControl.InheritanceFlags]::ObjectInherit
    $none = [Security.AccessControl.PropagationFlags]::None
    $allow = [Security.AccessControl.AccessControlType]::Allow
    $full = [Security.AccessControl.FileSystemRights]::FullControl
    $readExecute = [Security.AccessControl.FileSystemRights]::ReadAndExecute
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sidText in @('S-1-5-18', 'S-1-5-32-544', $installerSid)) {
        $sid = [Security.Principal.SecurityIdentifier]::new($sidText)
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
            $sid, $full, $inherit, $none, $allow
        )) | Out-Null
    }
    $users = [Security.Principal.SecurityIdentifier]::new('S-1-5-32-545')
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
        $users, $readExecute, $inherit, $none, $allow
    )) | Out-Null
    Set-Acl -LiteralPath $BridgeRoot -AclObject $acl
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
    # A direct grant does not remove an inherited or legacy explicit DENY ACE.
    # First restore ACL inheritance for this exact bridge directory and its
    # children, then immediately replace it with the narrowly scoped bridge
    # ACL below.  This remains contained to ProgramData\GX1\HostTelemetryBridge.
    $icacls = Join-Path $env:WINDIR 'System32\icacls.exe'
    Invoke-NativeChecked -FilePath $icacls -ArgumentList @(
        $BridgeRoot,
        '/reset', '/T', '/C'
    ) | Out-Null
    Set-BridgeDirectoryAcl -BridgeRoot $BridgeRoot
}

Assert-Administrator
Stop-ExistingBridgeTask -TaskName 'GX1HostTelemetryBridge'

$lhmRoot = Join-Path $env:ProgramData 'GX1\LibreHardwareMonitor\v0.9.6'
$lhmLibrary = Join-Path $lhmRoot 'LibreHardwareMonitorLib.dll'
if (-not (Test-Path -LiteralPath $lhmLibrary -PathType Leaf)) {
    throw "The verified LibreHardwareMonitor library is unavailable at $lhmLibrary. Run Install-GX1-HostTelemetry.ps1 first."
}
$nativeSmi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
if (-not (Test-Path -LiteralPath $nativeSmi -PathType Leaf)) {
    throw "Native nvidia-smi.exe is unavailable at $nativeSmi"
}

# Keep a damaged legacy bridge directory intact for forensic inspection.  An
# explicit versioned name allows a clean, elevated installation beside it when
# a historical ACL cannot be repaired without destructive intervention.
$bridgeRoot = Join-Path (Join-Path $env:ProgramData 'GX1') $BridgeDirectoryName
New-Item -ItemType Directory -Path $bridgeRoot -Force | Out-Null
Repair-ExistingBridgeDirectoryAccess -BridgeRoot $bridgeRoot
$configPath = Join-Path $bridgeRoot 'bridge-config.json'
$servicePath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeService.ps1'
$runnerPath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeRunner.ps1'
$serviceLogPath = Join-Path $bridgeRoot 'GX1-HostTelemetryBridgeService.log'
$publicCertificatePath = Join-Path $bridgeRoot 'GX1HostTelemetryBridgePublic.pem'
Wait-ForExclusiveFileAccess -Path $serviceLogPath
$loopbackEndpoint = "http://127.0.0.1:$Port/gx1/v1/telemetry/"
$wslListenAddress = ''
$wslClientAddressCanonical = ''
$wslEndpoint = ''
$legacyWslHttpEndpoint = ''
if (-not [string]::IsNullOrWhiteSpace($WslClientAddress)) {
    if ($WslPort -eq $Port) {
        throw 'WslPort must differ from the loopback bridge Port to prevent a Windows portproxy collision.'
    }
    $wslClientAddressCanonical = ConvertTo-CanonicalIpv4 -Value $WslClientAddress -Description 'WslClientAddress'
    $wslGateway = Get-WslGatewayIpv4
    $wslListenAddress = ConvertTo-CanonicalIpv4 -Value $wslGateway.IPAddress -Description 'Windows WSL gateway address'
    if (-not (Test-Ipv4SameSubnet -FirstAddress $wslListenAddress -SecondAddress $wslClientAddressCanonical -PrefixLength $wslGateway.PrefixLength)) {
        throw "WslClientAddress $wslClientAddressCanonical is not in the Windows WSL gateway subnet $wslListenAddress/$($wslGateway.PrefixLength)."
    }
    $wslEndpoint = "http://${wslListenAddress}:$WslPort/gx1/v1/telemetry/"
    $legacyWslHttpEndpoint = "http://${wslListenAddress}:$Port/gx1/v1/telemetry/"
}
$firewallRuleName = "GX1HostTelemetryBridge-Wsl-$WslPort"
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
$responseSchema = 'gx1_host_resource_telemetry_v2'
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
    Clear-CpuSensorValues -Computer $Computer
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

function Clear-CpuSensorValues {
    param([object]$Computer)
    foreach ($cpu in @($Computer.Hardware | Where-Object { $_.HardwareType.ToString() -eq 'Cpu' })) {
        foreach ($sensor in @($cpu.Sensors | Where-Object { $_.SensorType.ToString() -in @('Temperature', 'Power') })) {
            # Public LHM Sensor.Value cache only; no private state or MSR write.
            # Power/CoreMax otherwise retain old values on a failed update.
            $sensor.Value = $null
            if ($null -ne $sensor.Value) { throw 'CPU sensor cache could not be cleared' }
        }
    }
}

function Get-CpuResourceTelemetry {
    param([object]$Computer)
    $cpus = @($Computer.Hardware | Where-Object { $_.HardwareType.ToString() -eq 'Cpu' })
    if ($cpus.Count -ne 1 -or $cpus[0].Name -notmatch 'i9-10900K') {
        throw 'CPU identity did not match the qualified i9-10900K host'
    }
    $package = @($cpus[0].Sensors | Where-Object { $_.SensorType.ToString() -eq 'Temperature' -and $_.Name -eq 'CPU Package' })
    $cores = @($cpus[0].Sensors | Where-Object { $_.SensorType.ToString() -eq 'Temperature' -and $_.Name -match '^(CPU )?Core #[0-9]+$' })
    $power = @($cpus[0].Sensors | Where-Object { $_.SensorType.ToString() -eq 'Power' -and $_.Name -eq 'CPU Package' })
    if ($package.Count -ne 1 -or $cores.Count -ne 10 -or $power.Count -ne 1) {
        throw 'CPU package, ten physical core temperatures or package power sensor is missing'
    }
    foreach ($sensor in (@($package[0], $power[0]) + $cores)) {
        if ($null -eq $sensor.Value) { throw 'CPU sensor value is unavailable' }
        $value = [double]$sensor.Value
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0) {
            throw 'CPU sensor is non-finite or non-positive'
        }
    }
    return [pscustomobject]@{
        cpu_max_temp_c = [Math]::Max([double]$package[0].Value, [double](($cores | ForEach-Object { [double]$_.Value } | Measure-Object -Maximum).Maximum))
        cpu_package_power_w = [double]$power[0].Value
    }
}

function Get-HostResourceTelemetry {
    # Read real Windows counters; WSL /proc memory is a separate cgroup gate.
    $memory = Get-CimInstance Win32_PerfRawData_PerfOS_Memory -ErrorAction Stop
    foreach ($name in @('AvailableBytes', 'CommittedBytes', 'CommitLimit')) {
        if ($null -eq $memory.$name) { throw "Windows memory counter is unavailable: $name" }
    }
    $available = [uint64]$memory.AvailableBytes
    $used = [uint64]$memory.CommittedBytes
    $limit = [uint64]$memory.CommitLimit
    if ($limit -eq 0 -or $available -gt $script:HostPhysicalTotalBytes -or $used -gt $limit) {
        throw 'Windows memory counters violate physical bounds'
    }
    foreach ($value in @($available, $used, $limit, $script:HostPhysicalTotalBytes)) {
        if ($value -gt 9007199254740991) { throw 'Windows memory counter exceeds exact JSON integer range' }
    }
    $setup = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\Setup' -ErrorAction Stop
    foreach ($name in @('SystemSetupInProgress', 'OOBEInProgress')) {
        if ($null -eq $setup.$name -or $setup.$name -notin @(0, 1)) { throw 'Windows setup state is unknown' }
    }
    # Worker presence is a conservative proxy, including an idle worker. It is
    # not a claim to know every update phase or to prevent future OS restarts.
    $workers = @(Get-CimInstance Win32_Process -Filter "Name='TiWorker.exe' OR Name='MoUsoCoreWorker.exe' OR Name='SetupHost.exe' OR Name='SetupPrep.exe' OR Name='WindowsUpdateBox.exe' OR Name='wuauclt.exe'" -ErrorAction Stop)
    $cbs = Test-Path -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending' -ErrorAction Stop
    $wu = Test-Path -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired' -ErrorAction Stop
    $rebooting = Test-Path -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootInProgress' -ErrorAction Stop
    return [pscustomobject]@{
        host_physical_total_bytes = $script:HostPhysicalTotalBytes
        host_physical_available_bytes = $available
        host_commit_used_bytes = $used
        host_commit_limit_bytes = $limit
        cbs_reboot_pending = [bool]$cbs
        wu_reboot_required = [bool]$wu
        maintenance_active = [bool]($workers.Count -gt 0 -or $setup.SystemSetupInProgress -ne 0 -or $setup.OOBEInProgress -ne 0 -or $rebooting)
    }
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
$computer.IsCpuEnabled = $true
$computer.Open()
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
if ($null -eq $os.TotalVisibleMemorySize -or [uint64]$os.TotalVisibleMemorySize -eq 0) { throw 'Windows physical memory total is unavailable' }
$script:HostPhysicalTotalBytes = [uint64]$os.TotalVisibleMemorySize * 1024
$computer.Accept([Gx1HostTelemetryUpdateVisitor]::new())
Start-Sleep -Milliseconds 250
Clear-CpuSensorValues -Computer $computer
$computer.Accept([Gx1HostTelemetryUpdateVisitor]::new())
$null = Get-CpuResourceTelemetry -Computer $computer
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
            $cpu = Get-CpuResourceTelemetry -Computer $computer
            $hostResources = Get-HostResourceTelemetry
            $observed = [int64][Math]::Floor(
                ([System.Diagnostics.Stopwatch]::GetTimestamp() * 1000.0) / [System.Diagnostics.Stopwatch]::Frequency
            )
            $coreField = Format-CanonicalFloat -Value $native.core_temp_c
            $memoryField = Format-CanonicalFloat -Value $memory
            $drawField = Format-CanonicalFloat -Value $native.power_draw_w
            $limitField = Format-CanonicalFloat -Value $native.power_limit_w
            $cpuTempField = Format-CanonicalFloat -Value $cpu.cpu_max_temp_c
            $cpuPowerField = Format-CanonicalFloat -Value $cpu.cpu_package_power_w
            $cbsField = $hostResources.cbs_reboot_pending.ToString().ToLowerInvariant()
            $wuField = $hostResources.wu_reboot_required.ToString().ToLowerInvariant()
            $maintenanceField = $hostResources.maintenance_active.ToString().ToLowerInvariant()
            $payload = @(
                $responseSchema,
                $nonce,
                $native.uuid,
                $coreField,
                $memoryField,
                $drawField,
                $limitField,
                [string]($native.memory_used_mib),
                [string]$observed,
                $cpuTempField,
                $cpuPowerField,
                [string]$hostResources.host_physical_total_bytes,
                [string]$hostResources.host_physical_available_bytes,
                [string]$hostResources.host_commit_used_bytes,
                [string]$hostResources.host_commit_limit_bytes,
                $cbsField,
                $wuField,
                $maintenanceField
            ) -join "`n"
            $payload += "`n"
            $signature = Sign-BridgePayload -Rsa $rsa -Payload $payload
            $json = '{"schema_version":"' + $responseSchema + '","request_nonce":"' + $nonce + '","gpu_uuid":"' + $native.uuid + '","core_temp_c":' + $coreField + ',"memory_temp_c":' + $memoryField + ',"power_draw_w":' + $drawField + ',"power_limit_w":' + $limitField + ',"memory_used_mib":' + $native.memory_used_mib + ',"observed_monotonic_ms":' + $observed + ',"cpu_max_temp_c":' + $cpuTempField + ',"cpu_package_power_w":' + $cpuPowerField + ',"host_physical_total_bytes":' + $hostResources.host_physical_total_bytes + ',"host_physical_available_bytes":' + $hostResources.host_physical_available_bytes + ',"host_commit_used_bytes":' + $hostResources.host_commit_used_bytes + ',"host_commit_limit_bytes":' + $hostResources.host_commit_limit_bytes + ',"cbs_reboot_pending":' + $cbsField + ',"wu_reboot_required":' + $wuField + ',"maintenance_active":' + $maintenanceField + ',"signature":"'  + $signature + '"}'
            Write-BridgeResponse -Context $context -StatusCode 200 -Body $json
        }
        catch {
            Write-Warning ("telemetry-unavailable: " + $_.Exception.Message)
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
try {
    Add-Content -LiteralPath '$serviceLogPath' -Value 'runner-start'
    & '$servicePath' -CertificateThumbprint '$($certificate.Thumbprint)' -ExpectedGpuName '$ExpectedGpuName' -GpuIndex $GpuIndex -Port $Port *>> '$serviceLogPath'
    exit `$LASTEXITCODE
}
catch {
    Add-Content -LiteralPath '$serviceLogPath' -Value ("runner-fatal: " + (`$_ | Out-String))
    throw
}
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
    loopback_endpoint = $loopbackEndpoint
    wsl_endpoint = $wslEndpoint
    wsl_listen_address = $wslListenAddress
    wsl_client_address = $wslClientAddressCanonical
    wsl_proxy_port = if ([string]::IsNullOrWhiteSpace($wslEndpoint)) { $null } else { $WslPort }
    wsl_transport = if ([string]::IsNullOrWhiteSpace($wslEndpoint)) { '' } else { 'v4tov4_portproxy_to_windows_loopback' }
    service_path = $servicePath
    runner_path = $runnerPath
    service_log_path = $serviceLogPath
}
$configuration | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $configPath -Encoding UTF8
Export-PublicCertificatePem -Certificate $certificate -DestinationPath $publicCertificatePath
Set-BridgeDirectoryAcl -BridgeRoot $bridgeRoot

$netsh = Join-Path $env:WINDIR 'System32\netsh.exe'
# A first install has no URL ACL to remove; the loopback reservation is replaced
# either way.  A historical release attempted to give HTTP.sys the WSL gateway
# directly; remove only that exact obsolete reservation.  HTTP.sys keeps serving
# loopback, while the WSL transport below is an exact-address TCP port proxy.
& $netsh http delete urlacl "url=$loopbackEndpoint" 2>$null | Out-Null
if (-not [string]::IsNullOrWhiteSpace($legacyWslHttpEndpoint)) {
    & $netsh http delete urlacl "url=$legacyWslHttpEndpoint" 2>$null | Out-Null
}
Invoke-NativeChecked -FilePath $netsh -ArgumentList @('http', 'add', 'urlacl', "url=$loopbackEndpoint", 'user=NT AUTHORITY\SYSTEM') | Out-Null

# The WSL transport is opt-in.  Its TCP proxy is bound to the virtual WSL
# gateway (never 0.0.0.0) and forwards only to the already-owned Windows
# loopback listener. The firewall admits only the supplied WSL client address.
if (-not [string]::IsNullOrWhiteSpace($wslListenAddress)) {
    # Remove the same-port rule produced by the short-lived HTTP.sys transport
    # attempt before creating the distinct-port proxy below.
    & $netsh interface portproxy delete v4tov4 "listenaddress=$wslListenAddress" "listenport=$Port" protocol=tcp 2>$null | Out-Null
    & $netsh interface portproxy delete v4tov4 "listenaddress=$wslListenAddress" "listenport=$WslPort" protocol=tcp 2>$null | Out-Null
}
if (-not [string]::IsNullOrWhiteSpace($wslEndpoint)) {
    Invoke-NativeChecked -FilePath $netsh -ArgumentList @(
        'interface', 'portproxy', 'add', 'v4tov4',
        "listenaddress=$wslListenAddress",
        "listenport=$WslPort",
        'connectaddress=127.0.0.1',
        "connectport=$Port",
        'protocol=tcp'
    ) | Out-Null
}
Get-NetFirewallRule -DisplayName $firewallRuleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction Stop
if (-not [string]::IsNullOrWhiteSpace($wslEndpoint)) {
    $firewallArguments = @{
        DisplayName = $firewallRuleName
        Direction = 'Inbound'
        Action = 'Allow'
        Protocol = 'TCP'
        LocalAddress = $wslListenAddress
        LocalPort = $WslPort
        RemoteAddress = $wslClientAddressCanonical
        Profile = 'Any'
        EdgeTraversalPolicy = 'Block'
    }
    New-NetFirewallRule @firewallArguments | Out-Null
}

$taskName = 'GX1HostTelemetryBridge'
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
    loopback_endpoint = $loopbackEndpoint
    wsl_endpoint = $wslEndpoint
    wsl_client_address = $wslClientAddressCanonical
    wsl_proxy_port = if ([string]::IsNullOrWhiteSpace($wslEndpoint)) { $null } else { $WslPort }
    wsl_transport = if ([string]::IsNullOrWhiteSpace($wslEndpoint)) { '' } else { 'v4tov4_portproxy_to_windows_loopback' }
    public_certificate_windows_path = $publicCertificatePath
    public_certificate_wsl_path = "/mnt/c/ProgramData/GX1/$BridgeDirectoryName/GX1HostTelemetryBridgePublic.pem"
    public_certificate_sha256 = $certificateSha256
    gpu_uuid_expected_from_sensor_bootstrap = 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
    next_gate = 'Linux signed bridge probe against wsl_endpoint (when configured) and source binding; this does not authorize canonical training.'
}
Write-Host ''
Write-Host 'GX1 host telemetry bridge installed:' -ForegroundColor Green
$report | ConvertTo-Json -Depth 3
