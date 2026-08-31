$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "请先运行 scripts\setup.ps1 安装依赖。"
}
Set-Location (Join-Path $projectRoot "backend")
& $pythonPath -m app.worker
