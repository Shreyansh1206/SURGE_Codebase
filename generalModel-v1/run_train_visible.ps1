Set-Location $PSScriptRoot

Write-Host "Starting visible training (MiniGrid + Dino)..." -ForegroundColor Cyan
python train_visible.py @args
