$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = "siglip2"
$FrontendUrl = "http://localhost:5173/"

Write-Host "============================================"
Write-Host " SigLIP-2 Image Search App Launcher"
Write-Host "============================================"
Write-Host "Project: $RootDir"
Write-Host "Conda env: $EnvName"
Write-Host ""

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Cannot find conda in PATH." -ForegroundColor Red
    Write-Host "Please run this from Anaconda Prompt, or run: conda init"
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path (Join-Path $RootDir "backend\main.py"))) {
    Write-Host "[ERROR] backend\main.py not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path (Join-Path $RootDir "frontend\package.json"))) {
    Write-Host "[ERROR] frontend\package.json not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$BackendCmd = "cd /d `"$RootDir\backend`" && call conda activate $EnvName && python main.py"
$FrontendCmd = "cd /d `"$RootDir\frontend`" && npm run dev -- --host 0.0.0.0 --port 5173"

Write-Host "Starting backend..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $BackendCmd -WindowStyle Normal

Write-Host "Starting frontend..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $FrontendCmd -WindowStyle Normal

Write-Host "Waiting for services to start..."
Start-Sleep -Seconds 8

Write-Host "Opening browser: $FrontendUrl"
Start-Process $FrontendUrl

Write-Host ""
Write-Host "If the page shows network errors, wait until the backend finishes loading SigLIP-2 and building/loading indexes."
Read-Host "Press Enter to close this launcher window"
