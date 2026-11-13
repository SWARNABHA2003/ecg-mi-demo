@echo off
rem One-time setup on a Windows laptop (needs internet once). Requires Python 3.9-3.12 from python.org
rem with "Add python.exe to PATH" ticked during installation.
cd /d "%~dp0"
python -m venv .venv || (echo Python not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ecg_mi_app\requirements.txt
python ecg_mi_app\selftest.py
echo.
echo Setup finished. Start the demo by double-clicking start_demo.bat
pause
