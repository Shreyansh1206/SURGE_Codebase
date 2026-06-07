# Visible multi-task training — both MiniGrid and Dino on screen.
# Requires: pip install -r requirements.txt

Set-Location $PSScriptRoot

Write-Host "Starting visible training (MiniGrid + Dino)..." -ForegroundColor Cyan
python train_visible.py @args
