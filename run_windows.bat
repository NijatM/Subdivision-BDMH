@echo off
rem Double-click to launch the Hansmeyer subdivision explorer.
rem First run creates .venv and installs requirements; later runs start immediately.
setlocal
cd /d "%~dp0"
set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto deps

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY goto nopython
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul || goto nopython
echo Creating virtual environment...
%PY% -m venv .venv || goto fail

:deps
fc /b requirements.txt .venv\requirements.stamp >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies, first run only...
  "%VENV_PY%" -m pip install --upgrade pip >nul
  "%VENV_PY%" -m pip install -r requirements.txt || goto fail
  copy /y requirements.txt .venv\requirements.stamp >nul
)
"%VENV_PY%" app.py || goto fail
exit /b 0

:nopython
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click this file again.
pause
exit /b 1

:fail
echo Something went wrong - see the messages above.
pause
exit /b 1
