# ECG myocardial infarction screening (NORM / IMI / ASMI)

Web app that classifies a 12-lead ECG as **normal**, **inferior MI** or **anteroseptal MI** with an ensemble of
three dual-branch attention CNNs (SSPNet) trained on PTB-XL. It shows the median heartbeat of every lead, which
leads drove the decision, and the exact images the network saw.

**Live app:** <add your https://....streamlit.app link here>

## Contents
- `ecg_mi_app/app.py`: Streamlit app (entry point)
- `ecg_mi_app/ecg_preprocessing.py`, `ecg_mi_app/sspnet_model.py`: preprocessing and model, identical to the training notebook
- `runs/`: trained weights (3 seeds) and the ensemble configuration (class offsets)
- `ecg_mi_app/demo_examples/`: example ECGs from PTB-XL (not used for training)

## Run locally
`bash setup_linux_mac.sh` then `bash start_demo.sh` (Windows: `setup_windows.bat`, `start_demo.bat`).

## Deploy / update
1. `bash push_to_github.sh` (first time: creates the repository).
2. https://share.streamlit.io → *Create app* → this repository, branch `main`, main file `ecg_mi_app/app.py`,
   *Advanced settings* → Python 3.11 → *Deploy*.
3. Afterwards every `git push` (or `bash push_to_github.sh "message"`) redeploys automatically.

## Data
Example ECGs: PTB-XL v1.0.3 (Wagner et al., *Scientific Data* 7:154, 2020), PhysioNet, CC BY 4.0.

**Research prototype for a course project, not a medical device. Not for clinical use.**
