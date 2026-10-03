param([ValidateSet('prepare','masks','branch_prepare','vipe','track','geometry','variants','processor','infer')]
      [string]$Stage = 'prepare',
      [ValidateSet('A','B','C','all')][string]$Scene = 'all',
      [ValidateSet('smoke_vipe','pure_vipe','pure_vipe_unsmoothed','hybrid','paired_vipe','paired_hybrid')]
      [string]$Variant = 'pure_vipe',
      [ValidateSet('no_vda','default','rectified','sift_audited')][string]$Branch = 'no_vda')
$ErrorActionPreference = 'Stop'
$python = if ($Stage -in @('vipe','track','geometry')) {
    '/mnt/f/AIRI_task/.venv-vipe/bin/python'
} else {
    '/mnt/f/AIRI_task/.venv/bin/python'
}
wsl.exe -d Ubuntu --cd /mnt/f/AIRI_task/molmo-motion $python `
    scripts/dobbe_vipe_v1.py $Stage --scene $Scene --variant $Variant --branch $Branch
if ($LASTEXITCODE -ne 0) { throw "dobbe_vipe_v1 $Stage failed ($LASTEXITCODE)" }
