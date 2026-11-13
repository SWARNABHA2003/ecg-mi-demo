@echo off
cd /d "%~dp0"
set ECG_MI_PROJECT=%~dp0.
call .venv\Scripts\activate.bat
echo Starting ECG MI demo ... keep this window open; close it to stop the app.
python -m streamlit run ecg_mi_app\app.py --server.port 8501 --browser.gatherUsageStats false --theme.base light --theme.primaryColor "#B3122E"
pause
