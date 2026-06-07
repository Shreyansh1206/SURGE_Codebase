# Run 30-episode benchmark with live terminal output (no conda-run buffering).
# Usage (from dinoGame/ppo):  .\benchmarking\run_30ep.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$py = Join-Path $env:USERPROFILE "Miniconda3\envs\dinoGame\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "dinoGame env not found at $py — activate conda and run manually."
    exit 1
}

$out = "benchmarking/results/scratch_duck_best_duck_30_v2"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out "run.log"

Write-Host "Logging to $log"
$env:PYTHONUNBUFFERED = "1"

& $py -u benchmarking/benchmark.py `
    --ckpt checkpoints_scratch_duck/best_duck.pt `
    --episodes 30 `
    --headless `
    --out-dir $out 2>&1 | Tee-Object -FilePath $log
