$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$taskName='GX1RandomAccessCampaignV2'
$t=Get-ScheduledTask -TaskName $taskName
if(@($t.Actions).Count -ne 1 -or [string]$t.State -cne 'Running'){throw 'Expected one running campaign task'}
$oldController='C:\Users\Andre\GX1FullTrainNative_f77dd273\scripts\windows\GX1-RandomAccessCampaignV2Controller.ps1'
$launcher='C:\Users\Andre\GX1FullTrainNative_f77dd273\scripts\windows\GX1-NativeClockProfileLauncher.ps1'
$staged='C:\Users\Andre\GX1-NativeClockProfileLauncher.ps1'
$expected='40aa3b7f95b539e09bd1159bb5a50519afc2b1cc02bb4d0db3c2dfae3e427855'
if((Get-FileHash -LiteralPath $staged -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expected){throw 'Staged launcher changed'}
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($staged,[ref]$tokens,[ref]$errors)
if(@($errors).Count -ne 0){throw 'Launcher syntax invalid'}
$originalAst=[Management.Automation.Language.Parser]::ParseFile($oldController,[ref]$tokens,[ref]$errors)
if(@($errors).Count -ne 0 -or ($ast.ParamBlock.Extent.Text -cne $originalAst.ParamBlock.Extent.Text)){throw 'Controller parameter contract differs'}
$oldArguments=[string]$t.Actions[0].Arguments
if(-not $oldArguments.Contains('429a4e5db202375744cb961d40368dfed323984617175b8c37824d98da97816b') -or -not $oldArguments.Contains('"' + $oldController + '"')){throw 'Active campaign action differs'}
if((Test-Path -LiteralPath $launcher) -and (Get-FileHash -LiteralPath $launcher -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expected){throw 'Existing launcher differs'}
$root='C:\ProgramData\GX1\GpuClockProfile'
[void](New-Item -ItemType Directory -Path $root -Force)
$backup=Join-Path $root 'TASK_BEFORE_CLOCK_PROFILE.xml'
if(Test-Path -LiteralPath $backup){
 $backupXml=[xml][IO.File]::ReadAllText($backup)
 if([string]$backupXml.Task.Actions.Exec.Arguments -cne $oldArguments){throw 'Saved task action differs'}
}else{[IO.File]::WriteAllText($backup,(Export-ScheduledTask -TaskName $taskName),[Text.UTF8Encoding]::new($false))}
if(-not (Test-Path -LiteralPath $launcher)){Copy-Item -LiteralPath $staged -Destination $launcher}
$newArguments=$oldArguments.Replace('"' + $oldController + '"','"' + $launcher + '"')
$parameters=@{Execute=[string]$t.Actions[0].Execute;Argument=$newArguments}
if($t.Actions[0].WorkingDirectory){$parameters.WorkingDirectory=[string]$t.Actions[0].WorkingDirectory}
$action=New-ScheduledTaskAction @parameters
Set-ScheduledTask -TaskName $taskName -Action @($action)|Out-Null
$after=Get-ScheduledTask -TaskName $taskName
if(@($after.Actions).Count -ne 1 -or [string]$after.Actions[0].Arguments -cne $newArguments){throw 'Task update did not verify'}
$r=[ordered]@{decision='PASS_CLOCK_PROFILE_INSTALLED_FOR_NEXT_AUTOMATIC_START';observed_utc=[DateTimeOffset]::UtcNow.ToString('o');task_state=[string]$after.State;launcher=$launcher;launcher_sha256=$expected;original_controller_sha256='91a45151d108adccb0b2e5632993ae12f09e1baa6162ecc4092983bff062be25';plan_file_sha256='429a4e5db202375744cb961d40368dfed323984617175b8c37824d98da97816b';prior_task_xml=$backup;current_invocation_restarted=$false}
[IO.File]::WriteAllText((Join-Path $root 'INSTALL_RECEIPT.json'),($r|ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
$r|ConvertTo-Json -Compress
