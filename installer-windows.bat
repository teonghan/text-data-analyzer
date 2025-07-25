@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM --------------------------------------
REM Streamlit App Installer for Windows (with environment.yml)
REM --------------------------------------

SET ENV_NAME=textanalyzer
SET YAML_FILE=environment.yml
SET PYTHON_LAUNCHER=run_app.bat
SET SHORTCUT_NAME=Run Text Analyzer
SET THIS_DIR=%~dp0
SET SHORTCUT_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%.lnk
SET ICON_PATH=%THIS_DIR%icon.ico

REM Step 1: Check for Conda
where conda >nul 2>nul
IF ERRORLEVEL 1 (
    echo [ERROR] Conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda first.
    pause
    exit /b
)

REM Step 2: Check for environment.yml
IF NOT EXIST "%THIS_DIR%%YAML_FILE%" (
    echo [ERROR] environment.yml not found in %THIS_DIR%
    pause
    exit /b
)

REM Step 3: Create conda environment if it doesn't exist
CALL conda env list | findstr /C:"%ENV_NAME%" >nul
IF ERRORLEVEL 1 (
    echo Creating environment %ENV_NAME% from %YAML_FILE%...
    CALL conda env create -f "%THIS_DIR%%YAML_FILE%"
) ELSE (
    echo Environment %ENV_NAME% already exists.
)

REM Step 4: Create run_app.bat launcher
echo Creating launcher: %PYTHON_LAUNCHER%...

(
echo @echo off
echo CALL conda activate %ENV_NAME%
echo cd /d "%THIS_DIR%"
echo streamlit run app.py
echo pause
) > "%PYTHON_LAUNCHER%"

REM Step 5: Create Desktop Shortcut with Optional Icon
echo Creating desktop shortcut...
powershell -Command ^
  "$WshShell = New-Object -ComObject WScript.Shell; ^
   $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); ^
   $Shortcut.TargetPath = '%THIS_DIR%%PYTHON_LAUNCHER%'; ^
   $Shortcut.WorkingDirectory = '%THIS_DIR%'; ^
   if (Test-Path '%ICON_PATH%') { $Shortcut.IconLocation = '%ICON_PATH%' }; ^
   $Shortcut.Save()"

REM Done
echo.
echo ---------------------------------------------
echo ✅ Setup complete!
echo ➡ Shortcut: '%SHORTCUT_NAME%' added to your Desktop.
echo 📁 To launch the app, double-click the shortcut or run run_app.bat.
echo ---------------------------------------------
pause
