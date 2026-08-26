<#
.SYNOPSIS
  Installs and proves the host-only GPU sensor prerequisite for GX1.

.DESCRIPTION
  This script is intentionally a *sensor bootstrap*, not a canonical-training
  bypass.  It installs the pinned LibreHardwareMonitor release package and uses
  its library in this elevated, native Windows PowerShell process to require a
  numeric "GPU Memory Junction" value from the requested Nvidia GPU.

  It also queries Windows' native nvidia-smi for the GPU UUID and physical
  power limit.  The report is printed to the console only; a report produced by
  this script is not a canonical telemetry transport and cannot satisfy the
  signed bridge requirement on its own.

  Run only from *native Windows PowerShell as Administrator*, not from WSL.
  Example (after opening an elevated Windows PowerShell):
    Set-ExecutionPolicy -Scope Process Bypass -Force
    & '\\wsl.localhost\Ubuntu\home\andre2\src\GX1_ENGINE\scripts\windows\Install-GX1-HostTelemetry.ps1' -Install

  To set a physical power cap, opt in explicitly after the first sensor proof:
    ... -Install -SetPowerLimitWatts 250
#>

[CmdletBinding()]
param(
    [switch]$Install,
    [string]$ExpectedGpuName = 'NVIDIA GeForce RTX 3090',
    [ValidateRange(0, 31)]
    [int]$GpuIndex = 0,
    [ValidateRange(0, 1000)]
    [int]$SetPowerLimitWatts = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script from native Windows PowerShell as Administrator. No installation or probe was performed.'
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

function Get-PinnedLibreHardwareMonitorExe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallationRoot
    )

    $executable = Join-Path $InstallationRoot 'LibreHardwareMonitor.exe'
    $library = Join-Path $InstallationRoot 'LibreHardwareMonitorLib.dll'
    if ((Test-Path -LiteralPath $executable -PathType Leaf) -and
        (Test-Path -LiteralPath $library -PathType Leaf)) {
        return $executable
    }
    return $null
}

function Install-PinnedLibreHardwareMonitor {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallationRoot,
        [Parameter(Mandatory = $true)]
        [string]$ReleaseUri,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedSha256
    )

    $existing = Get-PinnedLibreHardwareMonitorExe -InstallationRoot $InstallationRoot
    if ($null -ne $existing) {
        return $existing
    }

    $parent = Split-Path -Parent $InstallationRoot
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    New-Item -ItemType Directory -Path $InstallationRoot -Force | Out-Null
    $temporaryZip = Join-Path $env:TEMP ("GX1-LibreHardwareMonitor-$PID.zip")
    try {
        Write-Host 'Downloading the pinned LibreHardwareMonitor release...'
        Invoke-WebRequest -Uri $ReleaseUri -OutFile $temporaryZip -UseBasicParsing
        $actualSha256 = (Get-FileHash -LiteralPath $temporaryZip -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($actualSha256 -ne $ExpectedSha256.ToUpperInvariant()) {
            throw "Refusing to install LibreHardwareMonitor: expected SHA-256 $ExpectedSha256, got $actualSha256."
        }
        Expand-Archive -LiteralPath $temporaryZip -DestinationPath $InstallationRoot -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryZip -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryZip -Force
        }
    }

    $installed = Get-PinnedLibreHardwareMonitorExe -InstallationRoot $InstallationRoot
    if ($null -eq $installed) {
        throw "Pinned LibreHardwareMonitor extraction did not produce its expected executable and library in $InstallationRoot"
    }
    return $installed
}

function Get-LhmMemoryJunctionProbe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath,
        [Parameter(Mandatory = $true)]
        [string]$RequiredGpuName
    )

    $libraryPath = Join-Path (Split-Path -Parent $ExecutablePath) 'LibreHardwareMonitorLib.dll'
    if (-not (Test-Path -LiteralPath $libraryPath -PathType Leaf)) {
        throw "LibreHardwareMonitorLib.dll was not found beside $ExecutablePath"
    }

    Add-Type -Path $libraryPath
    if ($null -eq ('Gx1LhmUpdateVisitor' -as [type])) {
        Add-Type -TypeDefinition @'
using LibreHardwareMonitor.Hardware;

public sealed class Gx1LhmUpdateVisitor : IVisitor
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
'@ -ReferencedAssemblies $libraryPath
    }

    $computer = [LibreHardwareMonitor.Hardware.Computer]::new()
    $computer.IsGpuEnabled = $true
    $computer.Open()
    try {
        $computer.Accept([Gx1LhmUpdateVisitor]::new())
        $gpus = @($computer.Hardware | Where-Object {
            $_.HardwareType.ToString() -eq 'GpuNvidia' -and $_.Name -eq $RequiredGpuName
        })
        if ($gpus.Count -ne 1) {
            $seen = @($computer.Hardware | Where-Object { $_.HardwareType.ToString() -eq 'GpuNvidia' } |
                ForEach-Object { $_.Name }) -join '; '
            throw "Expected exactly one Nvidia GPU named '$RequiredGpuName'; detected: $seen"
        }

        $junction = @($gpus[0].Sensors | Where-Object {
            $_.SensorType.ToString() -eq 'Temperature' -and $_.Name -eq 'GPU Memory Junction'
        })
        if ($junction.Count -ne 1 -or $null -eq $junction[0].Value) {
            $sensorNames = @($gpus[0].Sensors | Where-Object { $_.SensorType.ToString() -eq 'Temperature' } |
                ForEach-Object { "$($_.Name)=$($_.Value)" }) -join '; '
            throw "LibreHardwareMonitor did not provide one numeric GPU Memory Junction value. Temperature sensors seen: $sensorNames"
        }

        $value = [double]$junction[0].Value
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw "GPU Memory Junction was non-finite: $value"
        }
        return [pscustomobject]@{
            gpu_name = $gpus[0].Name
            memory_junction_c = [Math]::Round($value, 1)
        }
    }
    finally {
        $computer.Close()
    }
}

Assert-Administrator

$pinnedLhmRoot = Join-Path $env:ProgramData 'GX1\LibreHardwareMonitor\v0.9.6'
$pinnedLhmUri = 'https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/download/v0.9.6/LibreHardwareMonitor.zip'
$pinnedLhmSha256 = '086D9F1B5A99E643EDC2CFAAac16051685B551E4C5AC0B32A57C58C0E529C001'

if ($Install) {
    Install-PinnedLibreHardwareMonitor -InstallationRoot $pinnedLhmRoot -ReleaseUri $pinnedLhmUri -ExpectedSha256 $pinnedLhmSha256 | Out-Null
}

$lhmExe = Get-PinnedLibreHardwareMonitorExe -InstallationRoot $pinnedLhmRoot
if ($null -eq $lhmExe) {
    throw 'The pinned LibreHardwareMonitor release is not installed. Rerun with -Install.'
}

$nativeSmi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
if (-not (Test-Path -LiteralPath $nativeSmi -PathType Leaf)) {
    throw "Native nvidia-smi.exe was not found at $nativeSmi"
}

if ($SetPowerLimitWatts -gt 0) {
    Write-Host "Setting GPU index $GpuIndex physical power limit to $SetPowerLimitWatts W..."
    Invoke-NativeChecked -FilePath $nativeSmi -ArgumentList @('-i', "$GpuIndex", '-pl', "$SetPowerLimitWatts") | Out-Host
}

$rawGpu = @(Invoke-NativeChecked -FilePath $nativeSmi -ArgumentList @(
    '-i', "$GpuIndex", '--query-gpu=name,uuid,power.limit', '--format=csv,noheader,nounits'
))
if ($rawGpu.Count -ne 1) {
    throw "Expected exactly one GPU result at index $GpuIndex; got $($rawGpu.Count)."
}
$fields = @($rawGpu[0].Split(',') | ForEach-Object { $_.Trim() })
if ($fields.Count -ne 3 -or $fields[0] -ne $ExpectedGpuName -or [string]::IsNullOrWhiteSpace($fields[1])) {
    throw "Unexpected native nvidia-smi identity: '$($rawGpu[0])'. Expected GPU name '$ExpectedGpuName'."
}
$powerLimit = 0.0
if (-not [double]::TryParse($fields[2], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$powerLimit)) {
    throw "Native nvidia-smi returned a non-numeric power limit: '$($fields[2])'."
}

$sensor = Get-LhmMemoryJunctionProbe -ExecutablePath $lhmExe -RequiredGpuName $ExpectedGpuName
$report = [ordered]@{
    schema_version = 'gx1_host_telemetry_sensor_probe_v1'
    gpu_name = $fields[0]
    gpu_uuid = $fields[1]
    power_limit_w = [Math]::Round($powerLimit, 2)
    memory_junction_c = $sensor.memory_junction_c
    libre_hardware_monitor_exe = $lhmExe
    canonical_ready = ($powerLimit -le 250.0)
    note = 'Sensor-installation evidence only; this is not a signed canonical bridge response.'
}

Write-Host ''
Write-Host 'GX1 host sensor probe succeeded:' -ForegroundColor Green
$report | ConvertTo-Json -Depth 3
if (-not $report.canonical_ready) {
    Write-Warning "VRAM telemetry is now available, but the physical limit is $($report.power_limit_w) W. Canonical CUDA remains locked until it is at or below 250 W."
}
