param(
    [string]$ColmapExe = 'F:\AIRI_task\.tools\colmap-4.2.1\windows-cuda\bin\colmap.exe',
    [int]$UseGpu = 1,
    [string]$WslPython = '/mnt/f/AIRI_task/.venv/bin/python',
    [string]$AnalysisScriptName = 'dobbe_colmap_official.py',
    [string]$OutputName = 'dobbe_colmap_official_v1',
    [string[]]$Runs = @('full', 'causal'),
    [string]$CameraModel = 'SIMPLE_RADIAL',
    [string[]]$AnalysisArguments = @(),
    [switch]$SkipConditionalStride
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$outputRoot = Join-Path $repoRoot "runs\$OutputName"
$wslRepo = (wsl -d Ubuntu --exec wslpath -a $repoRoot).Trim()
$analysisScript = "$wslRepo/scripts/$AnalysisScriptName"

function Invoke-Analysis([string]$Mode) {
    wsl -d Ubuntu --exec $WslPython $analysisScript $Mode @AnalysisArguments
    if ($LASTEXITCODE -ne 0) { throw "Analysis mode $Mode failed: $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $ColmapExe)) { throw "COLMAP executable missing: $ColmapExe" }
$colmapDir = Split-Path $ColmapExe -Parent
$env:PATH = "$colmapDir;$colmapDir\..\lib;$env:PATH"
$env:QT_PLUGIN_PATH = "$colmapDir\..\plugins"
Invoke-Analysis 'prepare'
function Get-ColmapHelp([string[]]$Arguments, [string]$Prefix) {
    $stdOut = Join-Path $outputRoot "$Prefix.stdout.log"
    $stdErr = Join-Path $outputRoot "$Prefix.stderr.log"
    $helpProcess = Start-Process -FilePath $ColmapExe -ArgumentList $Arguments -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $stdOut -RedirectStandardError $stdErr
    if ($helpProcess.ExitCode -ne 0) { throw "COLMAP help failed: $Prefix" }
    return (Get-Content -LiteralPath $stdErr,$stdOut | Out-String)
}
$version = Get-ColmapHelp @('-h') 'version'
$helpText = Get-ColmapHelp @('automatic_reconstructor', '-h') 'automatic_help'
$version | Set-Content -LiteralPath (Join-Path $outputRoot 'colmap_version.txt') -Encoding UTF8
$helpText | Set-Content -LiteralPath (Join-Path $outputRoot 'automatic_reconstructor_help.txt') -Encoding UTF8

function Invoke-Reconstruction([string]$Slug) {
    $runPath = Join-Path $outputRoot $Slug
    $receiptPath = Join-Path $runPath 'execution.json'
    $cliArgs = @('automatic_reconstructor', '--workspace_path', $runPath,
        '--image_path', (Join-Path $runPath 'images'), '--data_type', 'video',
        '--single_camera', '1', '--camera_model', $CameraModel,
        '--quality', 'high', '--sparse', '1', '--dense', '0', '--use_gpu', "$UseGpu")
    if (Test-Path -LiteralPath $receiptPath) {
        $prior = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        if ($prior.exit_code -ne 0 -or (($prior.argv -join "`n") -ne ($cliArgs -join "`n"))) {
            throw "Existing run has incompatible or failed receipt: $Slug"
        }
    } else {
        Write-Output "START $Slug"
        $started = Get-Date
        $logPath = Join-Path $runPath 'colmap.log'
        $processArgs = $cliArgs | ForEach-Object { '"' + $_ + '"' }
        $process = Start-Process -FilePath $ColmapExe -ArgumentList $processArgs -WindowStyle Hidden -Wait -PassThru `
            -RedirectStandardError $logPath -RedirectStandardOutput (Join-Path $runPath 'colmap_stdout.log')
        $exitCode = $process.ExitCode
        @{argv=$cliArgs; executable=$ColmapExe; executable_sha256=(Get-FileHash -LiteralPath $ColmapExe -Algorithm SHA256).Hash.ToLower();
          version=$version; started_utc=$started.ToUniversalTime().ToString('o');
          elapsed_seconds=((Get-Date)-$started).TotalSeconds; exit_code=$exitCode} |
            ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
        if ($exitCode -ne 0) { throw "COLMAP $Slug failed with $exitCode; see $logPath" }
        Write-Output "COMPLETE $Slug"
    }
    Get-ChildItem -LiteralPath (Join-Path $runPath 'sparse') -Directory | ForEach-Object {
        $txtPath = Join-Path $runPath "sparse_txt\$($_.Name)"
        New-Item -ItemType Directory -Path $txtPath -Force | Out-Null
        $convertLog = Join-Path $txtPath 'converter.log'
        $convertArgs = @('model_converter', '--input_path', $_.FullName, '--output_path', $txtPath, '--output_type', 'TXT') |
            ForEach-Object { '"' + $_ + '"' }
        $convertProcess = Start-Process -FilePath $ColmapExe -ArgumentList $convertArgs -WindowStyle Hidden -Wait -PassThru `
            -RedirectStandardError $convertLog -RedirectStandardOutput (Join-Path $txtPath 'converter_stdout.log')
        $convertExit = $convertProcess.ExitCode
        if ($convertExit -ne 0) { throw "Model converter failed: $($_.Name)" }
    }
}

foreach ($runSlug in $Runs) { Invoke-Reconstruction $runSlug }
Invoke-Analysis 'analyze'
$decision = Get-Content -LiteralPath (Join-Path $outputRoot 'decision.json') -Raw | ConvertFrom-Json
if (-not $SkipConditionalStride -and $decision.conditional_stride_required) {
    Invoke-Analysis 'prepare-stride'
    Invoke-Reconstruction 'causal_stride5'
    Invoke-Analysis 'analyze'
}
Invoke-Analysis 'audit'
