"""
Builds ecg_mi_app/ptbxl_labels.csv: the cardiologist label of every PTB-XL record, so the app can show
"same as / differs from cardiologist" when a PTB-XL .hea/.dat file is uploaded. Run once:

    python3 ecg_mi_app/make_ptbxl_labels.py --ptbxl /path/to/ptb-xl/1.0.3

Columns: ecg_id, fold, label, codes
  label = NORM / IMI / ASMI when the record has exactly one diagnostic statement and it is one of these
          (the same rule as the training cohort); otherwise OTHER.
  codes = all diagnostic statements of the record (e.g. "IMI;LAFB"), for the app's note.
"""
import argparse
import ast
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ptbxl", required=True, help="PTB-XL folder with ptbxl_database.csv and scp_statements.csv")
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ptbxl_labels.csv"))
args = ap.parse_args()
root = Path(args.ptbxl)

db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
diagnostic = set(scp.index[scp.diagnostic == 1])
superclass = scp.diagnostic_class.to_dict()

rows = []
for eid, r in db.iterrows():
    codes = [c for c in ast.literal_eval(r.scp_codes) if c in diagnostic]
    supers = {superclass.get(c) for c in codes}
    label = codes[0] if (len(codes) == 1 and len(supers) == 1 and codes[0] in ("NORM", "IMI", "ASMI")) else "OTHER"
    rows.append((int(eid), int(r.strat_fold), label, ";".join(codes) if codes else "none"))

out = pd.DataFrame(rows, columns=["ecg_id", "fold", "label", "codes"])
out.to_csv(args.out, index=False)
print(f"Wrote {len(out)} records to {args.out}")
print(out.label.value_counts().to_string())
