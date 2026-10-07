@echo off
setlocal EnableExtensions
title ML Systems - Git setup
cd /d "%~dp0"

where git >nul 2>&1
if errorlevel 1 ( echo [X] Git not found. Run setup_env.bat first. & goto fail )

rem ---------- 1. Git identity ----------
echo [1/4] Checking Git identity ...
git config --global user.name >nul 2>&1
if not errorlevel 1 goto name_ok
set /p "GNAME=    Enter your full name for commits: "
git config --global user.name "%GNAME%"
:name_ok
git config --global user.email >nul 2>&1
if not errorlevel 1 goto email_ok
set /p "GEMAIL=    Enter your e-mail - use the one from your GitHub account: "
git config --global user.email "%GEMAIL%"
:email_ok
for /f "delims=" %%a in ('git config --global user.name') do echo     Name:  %%a
for /f "delims=" %%a in ('git config --global user.email') do echo     Email: %%a

rem ---------- 2. Init ----------
echo.
echo [2/4] Initialising repository ...
if exist ".git\" (
  echo     Repository already exists.
) else (
  git init -b main
  if errorlevel 1 goto fail
)

rem ---------- 3. Commit ----------
echo.
echo [3/4] Committing files ...
git add -A
git status --short
git diff --cached --quiet
if not errorlevel 1 ( echo     Nothing new to commit. & goto remote )
git commit -m "Initial commit: environment setup, pinned requirements, Lab 01 versions report"
if errorlevel 1 goto fail

rem ---------- 4. GitHub ----------
:remote
echo.
echo [4/4] GitHub ...
git remote get-url origin >nul 2>&1
if not errorlevel 1 goto push
echo     1. Open https://github.com/new
echo     2. Name it e.g. ml-systems-labs, choose Private,
echo        and do NOT add README, .gitignore or license.
echo     3. Click Create repository and copy its HTTPS URL.
echo.
set "REMOTE="
set /p "REMOTE=    Paste the repo URL, or just press Enter to skip: "
if not defined REMOTE ( echo     Skipped. Run this script again later to push. & goto done )
git remote add origin "%REMOTE%"
if errorlevel 1 goto fail
:push
echo     Pushing - a browser window may open to sign in to GitHub ...
git push -u origin main
if errorlevel 1 goto fail

:done
echo.
echo ==========================================================
echo  Done.
echo ==========================================================
git log --oneline -n 3
pause
exit /b 0

:fail
echo.
echo Stopped. Copy the error above and send it to Claude.
pause
exit /b 1
