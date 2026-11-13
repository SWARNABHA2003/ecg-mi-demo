"""
Copies a few PTB-XL example ECGs into ecg_mi_app/demo_examples/ so the demo runs offline and
without the PTB-XL folder. Run ONCE from the project folder, before the demo:

    python ecg_mi_app/prepare_demo.py --ptbxl /home/swarnabha/physionet.org/files/ptb-xl/1.0.3

Examples come from the TEST fold (10) if the final test evaluation was run (predictions.npz),
otherwise from the validation fold (9). The model never trained on either fold.
Per class: the most clearly recognised records ("showcase"), plus one inferior MI that the
model misses ("hard case"), so the demo also shows an honest limitation.
It also writes one ECG exported "like an ECG machine" (500 Hz, microvolts, other lead order)
to show that the app handles real-world files.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ecg_preprocessing as pp  # noqa: E402

CLASSES = ["NORM", "IMI", "ASMI"]

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=".")
ap.add_argument("--ptbxl", required=True)
ap.add_argument("--per-class", type=int, default=3)
args = ap.parse_args()
project, root = Path(args.project), Path(args.ptbxl)
ens = project / "runs" / "mi_sspnet_v5_ensemble"

if (ens / "predictions.npz").is_file():
    pred, fold = np.load(ens / "predictions.npz"), 10
else:
    pred, fold = np.load(ens / "ensemble_val_predictions.npz"), 9
    print("Test predictions not found -> using validation fold 9 examples.")
y, prob, ids = pred["y_true"], pred["prob"], pred["ecg_ids"]
yhat = prob.argmax(1)

db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
out = Path(__file__).resolve().parent / "demo_examples"
if out.exists():
    shutil.rmtree(out)
out.mkdir()

chosen = []
for k, c in enumerate(CLASSES):
    ok = np.where((y == k) & (yhat == k))[0]
    ok = ok[np.argsort(-prob[ok, k])][: args.per_class]
    chosen += [(i, "showcase") for i in ok]
miss = np.where((y == 1) & (yhat == 0))[0]
if len(miss):
    chosen.append((miss[np.argsort(-prob[miss, 0])][len(miss) // 2], "hard case"))

examples = []
for i, kind in chosen:
    eid = int(ids[i])
    rel = Path(db.loc[eid, "filename_lr"])
    for ext in (".hea", ".dat"):
        shutil.copy(root / (str(rel) + ext), out / (rel.name + ext))
    row = db.loc[eid]
    examples.append(dict(ecg_id=eid, record=rel.name, true_class=CLASSES[int(y[i])], fold=fold, kind=kind,
                         age=None if pd.isna(row.get("age")) else int(row["age"]),
                         sex=None if pd.isna(row.get("sex")) else ("male" if int(row["sex"]) == 0 else "female")))
    print(f"  {kind:9s} ecg_id {eid:5d}  true {CLASSES[int(y[i])]:4s}  model {CLASSES[int(yhat[i])]}")

# one "ECG machine" style export: 500 Hz, microvolts, different column order, time column
src = next(e for e in examples if e["true_class"] == "IMI" and e["kind"] == "showcase")
sig, fs, names = pp.read_wfdb(out / src["record"])
sig = pp.reorder_leads(sig, names)
sig500 = pp.resample_to_fs(sig, fs, 500) * 1000.0
cols = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III", "aVR", "aVL", "aVF"]
df = pd.DataFrame(sig500[:, [pp.LEADS.index(c) for c in cols]].round(1), columns=cols)
df.insert(0, "time", np.round(np.arange(len(df)) / 500, 4))
csv_name = f"ecg_machine_export_{src['ecg_id']}_500Hz_uV.csv"
df.to_csv(out / csv_name, index=False)

(out / "examples.json").write_text(json.dumps(dict(fold=fold, examples=examples, upload_example=dict(
    file=csv_name, ecg_id=src["ecg_id"], true_class="IMI", sampling_rate=500)), indent=2))
print(f"\nWrote {len(examples)} examples + {csv_name} to {out}")
