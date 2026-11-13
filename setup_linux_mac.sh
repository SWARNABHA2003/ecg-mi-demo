#!/usr/bin/env bash
# One-time setup on the laptop (needs internet once, ~1 GB download). Ubuntu: sudo apt install python3-venv
set -e
cd "$(dirname "$0")"
${PYTHON:-python3} -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then pip install torch; else pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
pip install -r ecg_mi_app/requirements.txt
python ecg_mi_app/selftest.py
bash ecg_mi_app/make_desktop_shortcut.sh || true
echo "Setup finished. Start the demo with:  bash start_demo.sh"
