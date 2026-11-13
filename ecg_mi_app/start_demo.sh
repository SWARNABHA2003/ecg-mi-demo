#!/usr/bin/env bash
# One-click start of the ECG MI demo app. Opens http://localhost:8501 in the browser.
# ---- edit ONE of these if your Python environment is not found automatically ----
CONDA_ENV=""          # e.g. CONDA_ENV="ecg"   (name of your conda environment)
VENV_PATH=""          # e.g. VENV_PATH="/home/swarnabha/Desktop/DH Project/.venv"
# -------------------------------------------------------------------------------
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$APP_DIR")"
cd "$PROJECT" || exit 1

if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then source "$VENV_PATH/bin/activate"
elif [ -n "$CONDA_ENV" ]; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
  conda activate "$CONDA_ENV"
elif [ -f "$PROJECT/.venv/bin/activate" ]; then source "$PROJECT/.venv/bin/activate"
elif [ -f "$PROJECT/venv/bin/activate" ]; then source "$PROJECT/venv/bin/activate"
fi

python -c "import torch, streamlit, wfdb" 2>/dev/null || {
  echo "This Python ($(command -v python)) is missing torch, streamlit or wfdb."
  echo "Set CONDA_ENV or VENV_PATH at the top of $APP_DIR/start_demo.sh, or run: pip install streamlit wfdb"
  read -rp "Press Enter to close." _; exit 1; }

export ECG_MI_PROJECT="$PROJECT"
echo "Starting ECG MI demo from $PROJECT ... (close this window to stop the app)"
python -m streamlit run "$APP_DIR/app.py" \
  --server.port 8501 --server.headless false --browser.gatherUsageStats false \
  --theme.base light --theme.primaryColor "#B3122E" --server.runOnSave false
