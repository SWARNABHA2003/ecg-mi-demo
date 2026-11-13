# ECG MI screening app (demo-ready)

A local browser app that runs the trained model (3-network ensemble + frozen class offsets) on a 12-lead ECG
and reports **NORM / inferior MI (IMI) / anteroseptal MI (ASMI)**, together with the evidence behind it:

- the result, its probabilities, the vote of each network, and ✓/✗ against the cardiologist's label when known;
- the **median heartbeat of every lead**, with the most influential leads in red;
- **lead importance**: how much the result changes when one lead is hidden (expected: II, III, aVF for IMI; V1–V4 for ASMI);
- the 12-lead ECG with detected beats, the exact input images given to the network, and quality warnings;
- a **How it works** tab for the audience and a downloadable JSON report.

It runs **fully offline** on your laptop. **Research prototype, not a medical device.**

---

## How your own model is connected

The app does not contain a model of its own. It loads **the networks you trained in the notebook**:

```
your ECG file ──► ecg_preprocessing.py ──► 2 × 12 images ──► sspnet_model.py ──► NORM / IMI / ASMI
 (CSV, WFDB, NPY)   same code as the        (signed beats +     SSPNet class (same code as the notebook)
                    notebook's cache        spectrograms)       + YOUR weights from runs/…/best.pt
```

- **Final ensemble (default):** reads `runs/mi_sspnet_v5_ensemble/ensemble_config.json`, which the notebook's
  Section 6b wrote. It lists your 3 trained runs and the class offsets. The app loads `best.pt` from each run,
  averages the 3 networks and applies the offsets. This is exactly the model in your report.
- **My own checkpoint file(s):** in the sidebar, choose *Trained model → My own checkpoint file(s)* and type the
  path of any `.pt` you trained, e.g. `runs/mi_sspnet_v5_s1/best.pt`, one per line. With several files the app
  averages them. A single file uses plain argmax (no offsets).

If you retrain later, nothing in the app changes: retrain in the notebook, rerun Section 6b, restart the app.
`selftest.py` proves the connection is correct: it compares the app's images and probabilities with the ones
the notebook saved.

## Model on another computer, demo on your laptop

**A. On the training system** (the one with `runs/`, the cache and PTB-XL). Unzip `ecg_mi_app` inside the project
folder, next to `runs/`, then:

```bash
cd "<project folder>"
python ecg_mi_app/export_bundle.py --ptbxl /path/to/ptb-xl/1.0.3
```

This writes **`ECG_MI_Demo.zip`**: your 3 `best.pt` files and their configs, the class offsets, the app, about 10
example ECGs and a self-test pack. It needs no Streamlit and no internet. Copy the zip to the laptop
(USB stick, Google Drive, …).

**B. On the laptop** (needs internet once, for about 1 GB of packages):

| Laptop | One-time setup | Start the demo |
|---|---|---|
| Ubuntu / macOS | `bash setup_linux_mac.sh` (Ubuntu first: `sudo apt install python3-venv`) | `bash start_demo.sh` or the desktop icon |
| Windows | install Python 3.11 from python.org (tick *Add to PATH*), then double-click `setup_windows.bat` | double-click `start_demo.bat` |

The setup ends with the self-test, which must print **RESULT: PASS**. After that, the laptop needs neither the
training system, nor PTB-XL, nor internet.

## One-time setup when the model is on the same computer

Unzip so that `ecg_mi_app/` sits **inside your project folder, next to `runs/`**, then in a terminal:

```bash
cd "/home/swarnabha/Desktop/DH Project"
source .venv/bin/activate          # or: conda activate <env>  (the environment the notebook uses)
pip install streamlit              # torch, wfdb, scipy, pandas, matplotlib are already there

# 1) check that the app reproduces the notebook exactly -> must end with "RESULT: PASS"
python ecg_mi_app/selftest.py --ptbxl /home/swarnabha/physionet.org/files/ptb-xl/1.0.3

# 2) copy example ECGs (test fold) into the app, so the demo needs neither PTB-XL nor internet
python ecg_mi_app/prepare_demo.py --ptbxl /home/swarnabha/physionet.org/files/ptb-xl/1.0.3

# 3) desktop icon for one-click start
bash ecg_mi_app/make_desktop_shortcut.sh
```

If the environment isn't a `.venv` / `venv` inside the project folder, open `ecg_mi_app/start_demo.sh` and set
`CONDA_ENV="..."` or `VENV_PATH="..."` at the top.

## Starting the app

- Double-click **ECG MI Demo** on the desktop. If Ubuntu asks, right-click → *Allow Launching* once.
- Or run `bash ecg_mi_app/start_demo.sh`.

The browser opens at http://localhost:8501. Keep the terminal window open; closing it stops the app.

## Inputs

| Mode | What it is |
|---|---|
| **Example ECGs** | the test-fold records copied by `prepare_demo.py`: ⭐ clearly recognised cases of each class, 🧩 one hard case (an inferior MI the model misses) |
| **Upload a recording** | WFDB (`.hea` + `.dat` together), CSV with lead names in the header, or NPY (samples × 12). `demo_examples/ecg_machine_export_*.csv` is an ECG exported "like a machine" (500 Hz, microvolts, different lead order) |
| **PTB-XL record** | any `ecg_id` from the PTB-XL folder; the app warns if it is from a training fold |

How the app adapts a real recording: it resamples to 100 Hz, converts microvolts to millivolts, reorders the
leads by name, and splits long recordings into 10-second windows whose probabilities are averaged.

## Files

```
ecg_mi_app/
├── start_demo.sh             one-click launcher (finds your Python environment)
├── make_desktop_shortcut.sh  creates the desktop icon
├── app.py                    the app (Streamlit)
├── ecg_preprocessing.py      file reading + the notebook's image code (unchanged)
├── sspnet_model.py           the notebook's SSPNet code (unchanged) + ensemble / decision rule
├── selftest.py               checks that the app reproduces the notebook
├── prepare_demo.py           copies example ECGs into demo_examples/
├── DEMO.md                   checklist and a 5-minute demo script
└── requirements.txt
```
