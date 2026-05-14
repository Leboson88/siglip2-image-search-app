@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "ENV_NAME=siglip2"
set "FRONTEND_URL=http://localhost:5173/"

echo ============================================
echo  SigLIP-2 Image Search App Launcher
echo ============================================
echo Project: %ROOT_DIR%
echo Conda env: %ENV_NAME%
echo.

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot find conda in PATH.
    echo Please open Anaconda Prompt once and run:
    echo   conda init
    echo Or edit this file and use your activate.bat path manually.
    pause
    exit /b 1
)

if not exist "%ROOT_DIR%backend\main.py" (
    echo [ERROR] backend\main.py not found.
    pause
    exit /b 1
)

if not exist "%ROOT_DIR%frontend\package.json" (
    echo [ERROR] frontend\package.json not found.
    pause
    exit /b 1
)

echo Starting backend...
start "SigLIP2 Backend" cmd /k "cd /d "%ROOT_DIR%backend" && call conda activate %ENV_NAME% && python main.py"

echo Starting frontend...
start "SigLIP2 Frontend" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev -- --host 0.0.0.0 --port 5173"

echo Waiting for services to start...
timeout /t 8 /nobreak >nul

echo Opening browser: %FRONTEND_URL%
start "" "%FRONTEND_URL%"

echo.
echo If the page shows network errors, wait until the backend finishes loading SigLIP-2 and building/loading indexes.
echo You can close this launcher window now. Backend and frontend are running in their own windows.
pause
