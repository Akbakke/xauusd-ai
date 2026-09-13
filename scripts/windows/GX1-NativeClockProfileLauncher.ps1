param(
    [Parameter(Mandatory = $true)][string]$PlanJson,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PlanFileSha256,
    [Parameter(Mandatory = $true)][string]$SourceRepo,
    [Parameter(Mandatory = $true)][Alias('WindowsSourceRepo')][string]$WindowsControllerSourceRoot,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExpectedControllerSha256,
    [string]$Distro = 'Ubuntu-22.04',
    [string]$LinuxUser = 'andre2',
    [string]$Python = '.venv/bin/python',
    [string]$CampaignControlRepo = $SourceRepo
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$controller = Join-Path $WindowsControllerSourceRoot 'scripts\windows\GX1-RandomAccessCampaignV2Controller.ps1'
if ((Get-FileHash -LiteralPath $controller -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ExpectedControllerSha256) {
    throw 'Bound campaign controller differs'
}
$controllerArguments = @{} + $PSBoundParameters
$gpu = 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29'
$clockLog = Join-Path $env:ProgramData 'GX1\GpuClockProfile'
[void](New-Item -ItemType Directory -Path $clockLog -Force)
$controllerExit = 0
try {
    # NVIDIA's WSL host-side clock controls. Do not change power limits,
    # numerical settings, signed telemetry or the canonical controller.
    $graphics = & nvidia-smi.exe -i $gpu -lgc 1395,1695
    if ($LASTEXITCODE -ne 0) { throw "GPU clock setup failed: $graphics" }
    $memory = & nvidia-smi.exe -i $gpu -lmc 9751
    if ($LASTEXITCODE -ne 0) { throw "GPU memory clock setup failed: $memory" }
    $record = [ordered]@{
        observed_utc = [DateTimeOffset]::UtcNow.ToString('o')
        boot_id = [long](Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters' -Name BootId).BootId
        plan_file_sha256 = $PlanFileSha256
        gpu_uuid = $gpu
        graphics_mhz = @(1395,1695)
        requested_memory_mhz = 9751
        controller_sha256 = $ExpectedControllerSha256
        launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    [IO.File]::AppendAllText((Join-Path $clockLog 'APPLIED.jsonl'), (($record | ConvertTo-Json -Depth 3 -Compress) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    & $controller @controllerArguments
    $controllerExit = $LASTEXITCODE
}
finally {
    # Normal completion/errors restore driver defaults; a physical reboot
    # also resets these locks before the next task invocation reapplies them.
    & nvidia-smi.exe -i $gpu -rgc | Out-Null
    & nvidia-smi.exe -i $gpu -rmc | Out-Null
}
exit $controllerExit
