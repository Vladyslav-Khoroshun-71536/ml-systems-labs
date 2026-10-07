@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ML Systems - environment setup

rem ============================================================
rem  Put this file in the ROOT of your course repo and run it.
rem  Python 3.11.8 is installed per-user and NOT added to PATH,
rem  so your Python 3.13 stays the default for other projects.
rem ============================================================

set "PYVER=3.11.8"
set "PYURL=https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
set "PYINST=%TEMP%\python-3.11.8-amd64.exe"
set "PYDIR=%LOCALAPPDATA%\Programs\Python\Python311"
set "PYEXE="

cd /d "%~dp0"
echo Working folder: %CD%
echo.

if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
  echo [WARN] ARM64 Windows detected. torch 2.3.1 has no ARM64 build - install will likely fail.
  pause
)

rem ---------- 1. Python 3.11.8 ----------
echo [1/5] Looking for Python %PYVER% ...
call :find_python
if defined PYEXE goto python_ok

echo     Not found. Downloading Python %PYVER% installer ...
curl -L --fail -o "%PYINST%" "%PYURL%"
if errorlevel 1 ( echo [X] Download failed. Check your internet connection. & goto fail )

echo     Installing per-user, without PATH changes - takes 1-2 minutes ...
start "" /wait "%PYINST%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0 AssociateFiles=0
set "INSTERR=%errorlevel%"
del "%PYINST%" >nul 2>&1

call :find_python
if not defined PYEXE (
  echo [X] Python %PYVER% not found after install - installer exit code !INSTERR!.
  echo     If a newer 3.11.x is already installed, uninstall it first and rerun.
  goto fail
)

:python_ok
echo     OK: %PYEXE%

rem ---------- 2. Git ----------
echo.
echo [2/5] Checking Git ...
where git >nul 2>&1
if not errorlevel 1 goto git_ok
echo     Git not found. Installing via winget ...
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
  echo [WARN] Git install failed. Install manually: https://git-scm.com/download/win
) else (
  echo     Git installed. Open a NEW cmd window afterwards so git is on PATH.
)
goto git_done
:git_ok
for /f "delims=" %%g in ('git --version') do echo     OK: %%g
:git_done

rem ---------- 3. Repo files ----------
echo.
echo [3/5] Checking repo files ...
if exist requirements.txt (
  echo     requirements.txt exists - using it.
) else (
  call :write_requirements
  echo     requirements.txt created.
)
if not exist .gitignore (
  call :write_gitignore
  echo     .gitignore created.
)
for %%d in (lab01\src lab01\results data) do if not exist "%%d" mkdir "%%d"
echo     Folders lab01\src, lab01\results, data are ready.

rem ---------- 4. Virtual environment ----------
echo.
echo [4/5] Checking virtual environment .venv ...
set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "%VPY%" goto make_venv
call :getver "%VPY%" VVER
if not "%VVER%"=="%PYVER%" (
  echo [X] .venv exists but uses Python %VVER%, not %PYVER%.
  echo     Delete the .venv folder and run this script again.
  goto fail
)
echo     OK: .venv uses Python %VVER%
goto venv_done
:make_venv
echo     Creating .venv with Python %PYVER% ...
"%PYEXE%" -m venv .venv
if errorlevel 1 ( echo [X] Failed to create .venv & goto fail )
:venv_done

rem ---------- 5. Packages ----------
echo.
echo [5/5] Installing packages from requirements.txt - torch is about 200 MB, be patient ...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo [X] pip install failed - see the errors above. & goto fail )
"%VPY%" -m pip check

rem ---------- Verification ----------
echo.
echo === Verification ===
"%VPY%" -c "import sys, numpy, pandas, sklearn, scipy, matplotlib, seaborn, torch, torchvision, onnx, onnxruntime, joblib, psutil; print('Python', sys.version.split()[0]); print('numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| torch', torch.__version__); print('All core imports OK')"
if errorlevel 1 ( echo [X] Some packages failed to import. & goto fail )

if exist "lab01\src\print_versions.py" (
  "!VPY!" lab01\src\print_versions.py > lab01\results\versions.txt
  echo Saved lab01\results\versions.txt
)

echo.
echo ==========================================================
echo  Done. Activate the environment with:
echo      .venv\Scripts\activate
echo ==========================================================
pause
exit /b 0

:fail
echo.
echo Setup stopped. Fix the error above and run the script again.
pause
exit /b 1

rem ============================================================
rem  Subroutines
rem ============================================================

:getver
rem  %1 = path to python.exe, %2 = output variable name
set "%~2="
if not exist "%~1" exit /b 0
for /f "tokens=2" %%v in ('call "%~1" --version 2^>^&1') do set "%~2=%%v"
exit /b 0

:find_python
set "PYEXE="
call :getver "%PYDIR%\python.exe" FV
if "%FV%"=="%PYVER%" ( set "PYEXE=%PYDIR%\python.exe" & exit /b 0 )
where py >nul 2>&1 || exit /b 0
set "FV="
for /f "tokens=2" %%v in ('py -3.11 --version 2^>nul') do set "FV=%%v"
if not "%FV%"=="%PYVER%" exit /b 0
for /f "delims=" %%p in ('py -3.11 -c "import sys; print(sys.executable)"') do set "PYEXE=%%p"
exit /b 0

:write_requirements
> requirements.txt (
  echo numpy==1.26.4
  echo pandas==2.2.2
  echo scikit-learn==1.5.0
  echo scipy==1.13.1
  echo matplotlib==3.9.0
  echo seaborn==0.13.2
  echo torch==2.3.1
  echo torchvision==0.18.1
  echo torchinfo==1.8.0
  echo thop==0.1.1.post2209072238
  echo onnx==1.16.1
  echo onnxruntime==1.18.0
  echo mlflow==2.14.1
  echo memory-profiler==0.61.0
  echo psutil==5.9.8
  echo codecarbon==2.4.2
  echo fastapi==0.111.0
  echo uvicorn==0.30.1
  echo pytest==8.2.2
  echo httpx==0.27.0
  echo locust==2.28.0
  echo requests==2.32.3
  echo pyarrow==15.0.2
  echo joblib==1.4.2
  echo tqdm==4.66.4
)
exit /b 0

:write_gitignore
> .gitignore (
  echo .venv/
  echo data/
  echo __pycache__/
  echo *.pyc
  echo .ipynb_checkpoints/
  echo .vscode/
  echo .idea/
)
exit /b 0
