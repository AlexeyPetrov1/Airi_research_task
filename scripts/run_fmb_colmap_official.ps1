param(
    [string]$ColmapExe = 'F:\AIRI_task\.tools\colmap-4.2.1\windows-cuda\bin\colmap.exe',
    [string]$WslPython = '/mnt/f/AIRI_task/.venv/bin/python',
    [ValidateSet('baseline', 'wrist-pinhole')]
    [string]$Profile = 'baseline'
)
$runSlugs = @(
    'episode2_side_1_full', 'episode2_side_1_causal',
    'episode3_side_1_full', 'episode3_side_1_causal',
    'episode2_wrist_1_full', 'episode2_wrist_1_causal',
    'episode3_wrist_1_full', 'episode3_wrist_1_causal'
)
if ($Profile -eq 'wrist-pinhole') {
    & (Join-Path $PSScriptRoot 'run_dobbe_colmap_official.ps1') `
        -ColmapExe $ColmapExe -WslPython $WslPython -CameraModel 'PINHOLE' `
        -AnalysisScriptName 'fmb_colmap_official.py' -OutputName 'fmb_colmap_official_pinhole_v1' `
        -AnalysisArguments @('--profile', 'wrist-pinhole') `
        -Runs @('episode2_wrist_1_causal', 'episode3_wrist_1_causal') -SkipConditionalStride
} else {
    & (Join-Path $PSScriptRoot 'run_dobbe_colmap_official.ps1') `
        -ColmapExe $ColmapExe -WslPython $WslPython `
        -AnalysisScriptName 'fmb_colmap_official.py' -OutputName 'fmb_colmap_official_v1' `
        -Runs $runSlugs -SkipConditionalStride
}
