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
    $h = (wsl.exe -- sh -lc 'printf %s "$HOME"').Trim()
    $p = "$h/src/GX1$([char]95)ENGINE/scripts/windows/Install-GX1-HostTelemetry.ps1"
    $script = (wsl.exe -- wslpath -w $p).Trim()
    & $script -Install

  To set a physical power cap, opt in explicitly after the first sensor proof:
    ... -Install -SetPowerLimitWatts 160

  This also installs the `GX1GpuPowerLimit` SYSTEM startup task.  The task
  waits for the Nvidia driver, reapplies the requested cap after every Windows
  restart or driver reset, and verifies the exact GPU UUID before succeeding.
#>

[CmdletBinding()]
param(
    [switch]$Install,
    [string]$ExpectedGpuName = 'NVIDIA GeForce RTX 3090',
    [ValidateRange(0, 31)]
    [int]$GpuIndex = 0,
    [ValidateRange(0, 160)]
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

function Install-PersistentPowerLimitTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$NativeSmi,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedGpuName,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedGpuUuid,
        [Parameter(Mandatory = $true)]
        [int]$GpuIndex,
        [Parameter(Mandatory = $true)]
        [int]$PowerLimitWatts
    )

    # Nvidia's `-pl` setting is a driver runtime property, not firmware.  It
    # can reset on a Windows restart, driver reset or power loss.  Keep this
    # tiny task separate from the signed telemetry bridge: the task changes
    # the host setting, while the bridge independently observes and signs it.
    $root = Join-Path $env:ProgramData 'GX1\GpuPowerLimit'
    $runnerPath = Join-Path $root 'GX1-ApplyGpuPowerLimit.ps1'
    $configPath = Join-Path $root 'GX1-GpuPowerLimit.config.json'
    $logPath = Join-Path $root 'GX1-GpuPowerLimit.log'
    $taskName = 'GX1GpuPowerLimit'
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $existingTask -and $existingTask.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
        foreach ($attempt in 1..25) {
            Start-Sleep -Milliseconds 200
            $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
            if ($null -eq $existingTask -or $existingTask.State -ne 'Running') {
                break
            }
        }
        if ($null -ne $existingTask -and $existingTask.State -eq 'Running') {
            throw "Existing $taskName task did not stop before its runner was replaced."
        }
    }
    New-Item -ItemType Directory -Path $root -Force | Out-Null

    $config = [ordered]@{
        schema_version = 'gx1_gpu_power_limit_startup_task_v1'
        native_smi = $NativeSmi
        expected_gpu_name = $ExpectedGpuName
        expected_gpu_uuid = $ExpectedGpuUuid
        gpu_index = $GpuIndex
        power_limit_w = $PowerLimitWatts
        initial_retry_count = 90
        retry_delay_seconds = 2
        recheck_seconds = 900
    }
    [System.IO.File]::WriteAllText(
        $configPath,
        ($config | ConvertTo-Json -Compress),
        [System.Text.UTF8Encoding]::new($false)
    )

    $runner = @'
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$configPath = Join-Path $root 'GX1-GpuPowerLimit.config.json'
$logPath = Join-Path $root 'GX1-GpuPowerLimit.log'
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json

function Write-Gx1PowerLimitLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    [System.IO.File]::AppendAllText($logPath, "$line`r`n", [System.Text.UTF8Encoding]::new($false))
}

$gpuIndex = [int]$config.gpu_index
$targetLimit = [int]$config.power_limit_w
$firstCheck = $true
while ($true) {
    $lastReason = 'Nvidia driver did not become ready.'
    $verified = $false
    $attemptLimit = if ($firstCheck) { [int]$config.initial_retry_count } else { 1 }
    foreach ($attempt in 1..$attemptLimit) {
        try {
            if (-not (Test-Path -LiteralPath ([string]$config.native_smi) -PathType Leaf)) {
                throw 'nvidia-smi.exe is unavailable'
            }
            $setOutput = @(& ([string]$config.native_smi) -i "$gpuIndex" -pl "$targetLimit" 2>&1)
            if ($LASTEXITCODE -ne 0) {
                throw "nvidia-smi -pl failed: $(($setOutput | Out-String).Trim())"
            }
            $raw = @(& ([string]$config.native_smi) -i "$gpuIndex" --query-gpu=name,uuid,power.limit --format=csv,noheader,nounits 2>&1)
            if ($LASTEXITCODE -ne 0 -or $raw.Count -ne 1) {
                throw 'nvidia-smi power-limit verification failed'
            }
            $fields = @($raw[0].ToString().Split(',') | ForEach-Object { $_.Trim() })
            $limit = 0.0
            if ($fields.Count -ne 3 -or
                $fields[0] -ne [string]$config.expected_gpu_name -or
                $fields[1] -ne [string]$config.expected_gpu_uuid -or
                -not [double]::TryParse($fields[2], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$limit) -or
                $limit -gt [double]$targetLimit) {
                throw "power-limit verification mismatch: '$($raw[0])'"
            }
            $verified = $true
            if ($firstCheck -or $attempt -gt 1) {
                Write-Gx1PowerLimitLog "SUCCESS gpu_uuid=$($fields[1]) power_limit_w=$limit attempt=$attempt"
            }
            break
        }
        catch {
            $lastReason = $_.Exception.Message
            if ($attempt -lt $attemptLimit) {
                Start-Sleep -Seconds ([int]$config.retry_delay_seconds)
            }
        }
    }
    if (-not $verified) {
        Write-Gx1PowerLimitLog "FAILURE $lastReason"
    }
    $firstCheck = $false
    Start-Sleep -Seconds ([int]$config.recheck_seconds)
}
'@
    [System.IO.File]::WriteAllText(
        $runnerPath,
        $runner,
        [System.Text.UTF8Encoding]::new($false)
    )

    $powerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runnerPath`""
    $action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName

    return [ordered]@{
        task_name = $taskName
        runner_path = $runnerPath
        log_path = $logPath
        configured_power_limit_w = $PowerLimitWatts
        startup_verification = 'SYSTEM retries for up to 180 seconds at boot, then reapplies and verifies the cap every 15 minutes using the exact GPU name and UUID.'
    }
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

$persistentPowerLimitTask = $null
if ($SetPowerLimitWatts -gt 0) {
    $persistentTaskParameters = @{
        NativeSmi = $nativeSmi
        ExpectedGpuName = $ExpectedGpuName
        ExpectedGpuUuid = $fields[1]
        GpuIndex = $GpuIndex
        PowerLimitWatts = $SetPowerLimitWatts
    }
    $persistentPowerLimitTask = Install-PersistentPowerLimitTask @persistentTaskParameters
}

$sensor = Get-LhmMemoryJunctionProbe -ExecutablePath $lhmExe -RequiredGpuName $ExpectedGpuName
$report = [ordered]@{
    schema_version = 'gx1_host_telemetry_sensor_probe_v1'
    gpu_name = $fields[0]
    gpu_uuid = $fields[1]
    power_limit_w = [Math]::Round($powerLimit, 2)
    memory_junction_c = $sensor.memory_junction_c
    libre_hardware_monitor_exe = $lhmExe
    canonical_ready = ($powerLimit -le 160.0)
    persistent_power_limit_task = $persistentPowerLimitTask
    note = 'Sensor-installation evidence only; this is not a signed canonical bridge response.'
}

Write-Host ''
Write-Host 'GX1 host sensor probe succeeded:' -ForegroundColor Green
$report | ConvertTo-Json -Depth 3
if (-not $report.canonical_ready) {
    Write-Warning "VRAM telemetry is now available, but the physical limit is $($report.power_limit_w) W. Canonical CUDA remains locked until it is at or below 160 W."
}
