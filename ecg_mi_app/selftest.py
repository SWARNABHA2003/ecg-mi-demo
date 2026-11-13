"""
Checks that the app reproduces the training notebook exactly.

On the laptop (after unzipping ECG_MI_Demo):     python ecg_mi_app/selftest.py
   uses ecg_mi_app/selftest_pack.npz (raw ECGs + the notebook's images and probabilities)
On the training system:                         python ecg_mi_app/selftest.py --ptbxl /path/to/ptb-xl/1.0.3
   reads raw PTB-XL records and the training cache directly

1. Preprocessing: raw ECG -> images, compared pixel by pixel with the notebook's images.
2. Model: ensemble probabilities, compared with the notebook's saved validation predictions.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import ecg_preprocessing as pp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=str(APP.parent), help="folder with runs/")
ap.add_argument("--ptbxl", default=None, help="PTB-XL folder (training system only)")
ap.add_argument("--n", type=int, default=30)
args = ap.parse_args()
project = Path(args.project)
ens_dir = project / "runs" / "mi_sspnet_v5_ensemble"
pack_file = APP / "selftest_pack.npz"

if args.ptbxl is None and pack_file.is_file():
    pk = np.load(pack_file, allow_pickle=False)
    names = [str(n) for n in pk["lead_names"]]
    raw = [(pk["signals"][k], float(pk["fs"][k]), names) for k in range(len(pk["ecg_ids"]))]
    ref_imgs, ref_raw, ref_final, ids = pk["images"], pk["prob_raw"], pk["prob"], pk["ecg_ids"]
else:
    import pandas as pd
    if args.ptbxl is None:
        sys.exit("No selftest_pack.npz found: pass --ptbxl /path/to/ptb-xl/1.0.3")
    cache = project / "ptbxl_cache_v4_signedbeat"
    meta = pd.read_csv(cache / "meta.csv")
    images = np.load(cache / "images.npy", mmap_mode="r")
    val = meta.index[meta.strat_fold == 9].to_numpy()
    pick = np.sort(np.random.default_rng(0).choice(val, size=min(args.n, len(val)), replace=False))
    raw = [pp.read_wfdb(Path(args.ptbxl) / meta.filename_lr[i]) for i in pick]
    ids = meta.ecg_id.to_numpy()[pick]
    ref_imgs = np.stack([images[i] for i in pick])
    saved = np.load(ens_dir / "ensemble_val_predictions.npz")
    pos = {int(e): k for k, e in enumerate(saved["ecg_ids"])}
    ref_raw = np.stack([saved["prob_raw"][pos[int(e)]] for e in ids])
    ref_final = np.stack([saved["prob"][pos[int(e)]] for e in ids])

print(f"[1] Preprocessing check on {len(ids)} validation ECGs")
bad, t0, app_imgs = 0, time.time(), []
for k, (sig, fs, names) in enumerate(raw):
    sig100, notes = pp.prepare_signal(sig, fs, names)
    img, _ = pp.ecg_to_v4_portraits(pp.windows_10s(sig100)[0][1])
    app_imgs.append(img)
    diff = int((img != ref_imgs[k]).sum())
    if diff:
        bad += 1
        print(f"    ecg_id {ids[k]}: {diff} pixels differ  notes={notes}")
print(f"    {len(ids) - bad}/{len(ids)} identical to the notebook  ({(time.time() - t0) / len(ids):.2f} s per ECG)")

print("[2] Model check against the notebook's saved predictions")
from sspnet_model import Ensemble  # noqa: E402
model = Ensemble(ens_dir, project)
t0 = time.time()
out = model.predict(np.stack(app_imgs))
dt = (time.time() - t0) / len(ids)
err = float(np.abs(out["prob"] - ref_raw).max())
agree = float((out["label"] == ref_final.argmax(1)).mean())
print(f"    max |probability difference| = {err:.2e}   same final label: {agree * 100:.0f}%   ({dt:.2f} s per ECG, {len(model.nets)} networks)")
ok = bad == 0 and err < 1e-4 and agree == 1.0
print("\nRESULT:", "PASS - the app reproduces the notebook." if ok else "CHECK the differences above.")
