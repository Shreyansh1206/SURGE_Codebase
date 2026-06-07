$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = if (Get-Command conda -ErrorAction SilentlyContinue) {
    @("conda", "run", "--no-capture-output", "-n", "dinoGame", "python", "-u")
} else {
    @("python", "-u")
}
& @py benchmark.py --ckpt checkpoints/best_duck.pt --episodes 30 --headless @args
