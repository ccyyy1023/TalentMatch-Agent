$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
if (-not (Test-Path -LiteralPath $venvPath)) {
    python -m venv $venvPath
}
& (Join-Path $venvPath "Scripts\python.exe") -m pip install -r (Join-Path $projectRoot "backend\requirements.txt")
Set-Location (Join-Path $projectRoot "frontend")
npm install
Write-Host "安装完成。请分别运行 scripts\start_backend.ps1 和 scripts\start_frontend.ps1。"
