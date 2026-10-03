param([string]$ColmapExe = 'F:\AIRI_task\.tools\colmap-4.2.1\windows-cuda\bin\colmap.exe')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$run = Join-Path $repo 'runs\dobbe_two_colmap_v1\official'
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/dobbe_two_colmap.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
$colmapDir = Split-Path $ColmapExe -Parent
$env:PATH = "$colmapDir;$colmapDir\..\lib;$env:PATH"
$env:QT_PLUGIN_PATH = "$colmapDir\..\plugins"
$cliArgs = @('automatic_reconstructor', '--workspace_path', $run, '--image_path', (Join-Path $run 'images'),
    '--data_type', 'video', '--single_camera', '1', '--camera_model', 'PINHOLE', '--quality', 'high',
    '--sparse', '1', '--dense', '0', '--use_gpu', '1')
$receipt = Join-Path $run 'execution.json'
if (Test-Path -LiteralPath $receipt) {
    $prior = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
    if ($prior.exit_code -ne 0 -or ($prior.argv -join "`n") -ne ($cliArgs -join "`n")) { throw 'Existing receipt differs or failed' }
} else {
    if (Test-Path -LiteralPath (Join-Path $run 'database.db')) { throw 'Unreceipted DB exists: need a clean workspace' }
    $started = Get-Date
    $quoted = $cliArgs | ForEach-Object { '"' + $_ + '"' }
    $proc = Start-Process -FilePath $ColmapExe -ArgumentList $quoted -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardError (Join-Path $run 'colmap.log') -RedirectStandardOutput (Join-Path $run 'colmap_stdout.log')
    @{argv=$cliArgs; executable=$ColmapExe; executable_sha256=(Get-FileHash -LiteralPath $ColmapExe -Algorithm SHA256).Hash.ToLower();
      started_utc=$started.ToUniversalTime().ToString('o'); elapsed_seconds=((Get-Date)-$started).TotalSeconds;
      exit_code=$proc.ExitCode; clean_database=$true} | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $receipt -Encoding UTF8
    if ($proc.ExitCode -ne 0) { throw "Official COLMAP failed: $($proc.ExitCode)" }
}
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion /mnt/f/AIRI_task/.venv/bin/python scripts/dobbe_two_colmap.py inputs
if ($LASTEXITCODE -ne 0) { throw 'Input construction failed' }
