param(
    [int]$Episodes = 240,
    [int]$Seed = 1000,
    [string]$Dataset = 'data/mujoco_jenga',
    [string]$Output = 'checkpoints/mujoco_vjepa'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $Python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
    & $Python -m src.sim.mujoco_jenga --episodes $Episodes --seed $Seed --output $Dataset
    if ($LASTEXITCODE -ne 0) { throw 'MuJoCo generation failed' }
    & $Python scripts/extract_vjepa_features.py --dataset $Dataset --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Pretrained feature extraction failed' }
    & $Python -m src.parallel.train_mujoco_risk --directory $Output
    if ($LASTEXITCODE -ne 0) { throw 'Risk-head training failed' }
    & $Python -m src.parallel.train_motion_student --dataset $Dataset --directory $Output
    if ($LASTEXITCODE -ne 0) { throw 'Motion student training failed' }
    & $Python -m src.parallel.mujoco_demo --dataset $Dataset --directory $Output
    if ($LASTEXITCODE -ne 0) { throw 'Parallel inference check failed' }
} finally {
    Pop-Location
}
