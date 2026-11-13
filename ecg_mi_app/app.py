"""
ECG MI Screening: normal / inferior MI / anteroseptal MI from a 12-lead ECG.
Simple three-step interface: 1 choose an ECG, 2 result, 3 why.

Start with ecg_mi_app/start_demo.sh, or from the project folder (the one that contains runs/):
    streamlit run ecg_mi_app/app.py
"""
import ast
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ecg_preprocessing as pp

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "demo_examples"
CLASS_NAMES = ["NORM", "IMI", "ASMI"]
NICE = {"NORM": "Normal", "IMI": "Inferior MI", "ASMI": "Anteroseptal MI"}
VERDICT = {
    "NORM": ("Normal ECG", "✅",
             "No sign of a heart attack in the lower (inferior) or front-centre (anteroseptal) wall of the heart."),
    "IMI": ("Inferior MI detected", "⚠️",
            "Pattern typical of a heart attack (myocardial infarction) in the <b>lower wall</b> of the heart, "
            "seen in leads II, III and aVF."),
    "ASMI": ("Anteroseptal MI detected", "⚠️",
             "Pattern typical of a heart attack (myocardial infarction) in the <b>front-centre wall</b> of the heart, "
             "seen in leads V1 to V4."),
}
EXPECTED = {"IMI": {"II", "III", "aVF"}, "ASMI": {"V1", "V2", "V3", "V4"}}
GREEN, RED, INK, MUTED = "#1E7B53", "#B3122E", "#1B1F2A", "#5A6170"
PAPER_MINOR, PAPER_MAJOR = "#F6D5D9", "#EBA9B2"

st.set_page_config(page_title="ECG MI Screening", page_icon="❤️", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button {font-family: 'Inter', sans-serif;}
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.stApp {background: #F5F6F8;}
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 860px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #FFFFFF; border-radius: 16px !important;
  border: 1px solid #E6E8EC !important; box-shadow: 0 1px 3px rgba(16,24,40,.06);}
.step {display:flex; align-items:center; gap:10px; font-size:19px; font-weight:600; color:#1B1F2A; margin:2px 0 8px 0;}
.step span {display:inline-flex; width:28px; height:28px; border-radius:50%; background:#B3122E; color:#fff;
  align-items:center; justify-content:center; font-size:15px; font-weight:700;}
.hint {color:#5A6170; font-size:14.5px; margin:-4px 0 10px 0;}
.chip {display:inline-block; padding:4px 12px; border-radius:999px; font-size:13.5px; margin:8px 8px 0 0;
  background:#F1F3F6; color:#1B1F2A;}
.why {font-size:16px; color:#1B1F2A; margin:4px 0 8px 0;}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#C9CDD4", "xtick.color": MUTED, "ytick.color": MUTED})


# ------------------------------------------------------------------------------------------- cached work
@st.cache_resource(show_spinner="Loading the trained network(s)...")
def load_model(ensemble_dir, runs_root, threads, checkpoints=None):
    from sspnet_model import Ensemble
    if checkpoints:
        return Ensemble(checkpoints=list(checkpoints), threads=threads)
    return Ensemble(ensemble_dir, runs_root, threads)


@st.cache_data(show_spinner=False)
def load_ptbxl_db(root):
    db = pd.read_csv(Path(root) / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    return db


@st.cache_data(show_spinner=False)
def load_examples():
    f = EXAMPLES_DIR / "examples.json"
    return json.loads(f.read_text()) if f.is_file() else None


@st.cache_data(show_spinner=False)
def load_ptbxl_labels():
    f = APP_DIR / "ptbxl_labels.csv"
    return pd.read_csv(f, index_col="ecg_id") if f.is_file() else None


def ptbxl_reference(filename):
    """Cardiologist reference for a PTB-XL file name such as 04209_lr.hea / 04209_hr.dat / ..._4209_....csv."""
    import re
    labels = load_ptbxl_labels()
    if labels is None:
        return None
    m = re.match(r"^(\d{5})_(lr|hr)$", Path(filename).stem)
    if not m:
        return None
    eid = int(m.group(1))
    if eid not in labels.index:
        return None
    row = labels.loc[eid]
    return dict(ecg_id=eid, label=row["label"], codes=str(row["codes"]).replace(";", ", "), fold=int(row["fold"]))


@st.cache_data(show_spinner=False, max_entries=64)
def analyse(_model, model_key, sig100, explain):
    """Everything the page shows, computed once per recording (switching examples is instant afterwards)."""
    t0 = time.time()
    wins = pp.windows_10s(sig100)
    imgs, _ = zip(*[pp.ecg_to_v4_portraits(w) for _, w in wins])
    imgs = np.stack(imgs)
    out = _model.predict(imgs)
    p_rec = out["prob"].mean(0, keepdims=True)
    lab, p_adj = _model.decide(p_rec)
    lab = int(lab[0])
    best_w = int(np.argmax(out["prob_adjusted"][:, lab]))
    sig10 = wins[best_w][1]
    beat, peaks, nb = pp.median_beat_mv(sig10)
    imp = _model.lead_importance(imgs[best_w], lab) if explain else None
    return dict(label=lab, p_rec=p_rec[0], p_adj=p_adj[0], member=out["member_probs"].mean(1),
                win_starts=[s for s, _ in wins], win_prob=out["prob_adjusted"], win_label=out["label"],
                best_w=best_w, sig10=sig10, img=imgs[best_w], beat=beat, peaks=peaks, nb=nb,
                flags=pp.quality_flags(sig10, nb), imp=imp, seconds=time.time() - t0)

# ------------------------------------------------------------------------------------------- plots (ECG-paper style)
def _paper(ax, xmin, xmax, ymin, ymax, minor_x, major_x, minor_y=0.1, major_y=0.5):
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(xmin, xmax + 1e-9, major_x)); ax.set_yticks(np.arange(ymin, ymax + 1e-9, major_y))
    ax.set_xticks(np.arange(xmin, xmax + 1e-9, minor_x), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1e-9, minor_y), minor=True)
    ax.grid(which="minor", color=PAPER_MINOR, lw=0.4); ax.grid(which="major", color=PAPER_MAJOR, lw=0.7)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_heartbeats(beat, key_leads=()):
    fig, axes = plt.subplots(3, 4, figsize=(10, 5.4))
    t = (np.arange(len(beat)) - pp.PRE) * 1000 / pp.FS
    order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]          # classic 3 x 4 ECG layout
    for k, ax in enumerate(axes.flat):
        l = order[k]
        hi = pp.LEADS[l] in key_leads
        _paper(ax, -300, 600, -1.5, 1.5, 40, 200)
        ax.plot(t, beat[:, l], color=RED if hi else INK, lw=1.8 if hi else 1.3)
        ax.text(-285, 1.15, pp.LEADS[l], fontsize=11, fontweight="bold", color=RED if hi else INK)
    fig.tight_layout(pad=0.6, h_pad=0.6, w_pad=0.6)
    return fig


def plot_strip(sig10, lead):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    t = np.arange(len(sig10)) / pp.FS
    _paper(ax, 0, 10, -2, 2, 0.04 * 5, 1.0)
    ax.plot(t, sig10[:, lead], color=INK, lw=1.0)
    ax.text(0.08, 1.55, pp.LEADS[lead], fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(pad=0.3)
    return fig


def plot_importance(imp):
    fig, ax = plt.subplots(figsize=(9, 2.4))
    top = set(np.argsort(imp)[-3:])
    ax.bar(pp.LEADS, imp, color=[RED if i in top else "#C3C8D0" for i in range(12)])
    ax.axhline(0, color=MUTED, lw=0.6); ax.set_ylabel("Influence")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_model_input(img):
    fig, axes = plt.subplots(2, 12, figsize=(12, 2.4))
    for b in range(2):
        for l in range(12):
            ax = axes[b, l]
            ax.imshow(img[b, l], cmap="gray" if b == 0 else "magma", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if b == 0:
                ax.set_title(pp.LEADS[l], fontsize=8)
    fig.tight_layout(pad=0.3)
    return fig


COMMON_FS = [100, 125, 128, 200, 250, 256, 300, 360, 400, 500, 512, 1000, 2000]


def detect_fs(n_samples, file_bytes=None):
    """Sampling rate of a CSV/NPY ECG: from a time column if there is one, otherwise assume a
    standard 10-s recording and snap to the nearest common ECG rate. Returns (fs, how)."""
    if file_bytes is not None:
        try:
            import io
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
            tcol = [c for c in df.columns if str(c).strip().lower() in ("time", "t", "time (s)", "time_s", "seconds")]
            if tcol:
                dt = float(np.median(np.diff(df[tcol[0]].to_numpy(dtype=float))))
                if dt > 0:
                    fs = 1.0 / dt
                    if fs > 5000:          # time column in milliseconds
                        fs /= 1000.0
                    if dt >= 1:            # e.g. milliseconds 0, 2, 4, ...
                        fs = 1000.0 / dt
                    return float(round(fs)), "from the time column"
        except Exception:  # noqa: BLE001
            pass
    guess = n_samples / 10.0
    near = min(COMMON_FS, key=lambda f: abs(f - guess))
    fs = float(near if abs(near - guess) / guess < 0.06 else round(guess))
    return fs, "assuming a 10-second recording"

def choose(label, options, default, key, fmt=None):
    """Clickable chips if this Streamlit has them, otherwise horizontal radio buttons."""
    fmt = fmt or (lambda x: x)
    if hasattr(st, "pills"):
        v = st.pills(label, options, default=default, format_func=fmt, key=key, label_visibility="collapsed")
        return default if v is None else v
    return st.radio(label, options, index=options.index(default), format_func=fmt, key=key,
                    horizontal=True, label_visibility="collapsed")

def read_uploads(files, fs_override):
    """Group uploaded files into recordings (.hea+.dat pairs by name, each CSV/NPY on its own).
    Returns (records, problems)."""
    groups = {}
    for f in files:
        p = Path(f.name)
        groups.setdefault(p.stem, {})[p.suffix.lower()] = f
    recs, problems = [], []
    for stem, g in sorted(groups.items()):
        try:
            if ".hea" in g or ".dat" in g or ".mat" in g or ".xyz" in g:
                if ".hea" not in g or not ({".dat", ".mat"} & set(g)):
                    problems.append(f"{stem}: a WFDB recording needs its .hea file and its signal file "
                                    f"(.dat or .mat; PTB-DB also .xyz), skipped.")
                    continue
                with tempfile.TemporaryDirectory() as td:
                    for f in g.values():
                        (Path(td) / f.name).write_bytes(f.getvalue())
                    sig, fs, names = pp.read_wfdb(Path(td) / stem)
                rec = dict(sig=sig, fs=fs, names=names, source=g[".hea"].name, fs_note=None)
            elif ".csv" in g or ".txt" in g:
                f = g.get(".csv") or g[".txt"]
                sig, names = pp.read_table(f.getvalue(), f.name)
                fs, how = detect_fs(len(sig), f.getvalue())
                if fs_override != "Auto":
                    fs, how = float(fs_override), "set in Settings"
                rec = dict(sig=sig, fs=fs, names=names, source=f.name, fs_note=f"Sampling rate {fs:g} Hz ({how}).")
            elif ".npy" in g:
                f = g[".npy"]
                sig = np.load(f, allow_pickle=False)
                n_ = sig.shape[0] if sig.ndim == 2 and sig.shape[1] == 12 else sig.shape[-1]
                fs, how = detect_fs(n_)
                if fs_override != "Auto":
                    fs, how = float(fs_override), "set in Settings"
                rec = dict(sig=sig, fs=fs, names=None, source=f.name, fs_note=f"Sampling rate {fs:g} Hz ({how}).")
            else:
                continue
        except Exception as ex:  # noqa: BLE001
            problems.append(f"{stem}: could not be read ({ex}).")
            continue
        ref = ptbxl_reference(rec["source"])
        truth = None
        if examples and examples.get("upload_example") and Path(rec["source"]).name == examples["upload_example"]["file"]:
            truth = examples["upload_example"]["true_class"]
        if ref is not None:
            rec["source"] = f"PTB-XL #{ref['ecg_id']}"
            truth = ref["label"] if ref["label"] in CLASS_NAMES else None
        rec.update(ref=ref, truth=truth)
        recs.append(rec)
    return recs, problems


def step(n, title, hint=None):
    st.markdown(f"<div class='step'><span>{n}</span>{title}</div>" + (f"<div class='hint'>{hint}</div>" if hint else ""),
                unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- settings (hidden sidebar)
examples = load_examples()
PTBXL_DEFAULT = os.environ.get("ECG_MI_PTBXL", "/home/swarnabha/physionet.org/files/ptb-xl/1.0.3")
default_project = Path(os.environ.get("ECG_MI_PROJECT", APP_DIR.parent))
st.sidebar.markdown("### ⚙️ Settings")
explain = True
with st.sidebar.expander("Model"):
    project = Path(st.text_input("Project folder (contains runs/)", value=str(default_project)))
    model_src = st.radio("Trained model", ["Final ensemble (3 networks)", "My own checkpoint file(s)"])
    ens_dir = str(project / "runs" / "mi_sspnet_v5_ensemble")
    ckpts = None
    if model_src.startswith("Final"):
        ens_dir = st.text_input("Ensemble folder", value=ens_dir)
    else:
        txt = st.text_area("Checkpoint path(s), one per line", value=str(project / "runs" / "mi_sspnet_v5_s1" / "best.pt"))
        ckpts = tuple(p.strip() for p in txt.splitlines() if p.strip())
    fs_override = st.selectbox("Sampling rate of CSV / NPY uploads", ["Auto", 100, 250, 500, 1000])
    ncpu = os.cpu_count() or 2
    threads = st.slider("CPU threads", 1, max(1, ncpu), min(4, ncpu))
    verify = st.button("Verify model", help="Recomputes 20 validation ECGs and compares with the training notebook.")

try:
    model = load_model(ens_dir, str(project), threads, ckpts)
except Exception as e:  # noqa: BLE001
    st.error(f"The trained model could not be loaded.\n\n{e}\n\nOpen the sidebar (›) → Settings → Model and check the project folder.")
    st.stop()
model_key = f"{ens_dir}|{ckpts}|{threads}"

if verify:
    pack = APP_DIR / "selftest_pack.npz"
    if not pack.is_file():
        st.sidebar.error("selftest_pack.npz not found.")
    else:
        with st.spinner("Verifying..."):
            pk = np.load(pack, allow_pickle=False)
            names_ = [str(n) for n in pk["lead_names"]]
            imgs_, bad_ = [], 0
            for k_ in range(len(pk["ecg_ids"])):
                s_, _ = pp.prepare_signal(pk["signals"][k_], float(pk["fs"][k_]), names_)
                im_, _ = pp.ecg_to_v4_portraits(pp.windows_10s(s_)[0][1])
                bad_ += int(not np.array_equal(im_, pk["images"][k_]))
                imgs_.append(im_)
            o_ = model.predict(np.stack(imgs_))
            err_ = float(np.abs(o_["prob"] - pk["prob_raw"]).max())
            agree_ = float((o_["label"] == pk["prob"].argmax(1)).mean())
        n_ = len(pk["ecg_ids"])
        msg_ = f"images {n_ - bad_}/{n_} identical · max difference {err_:.1e} · labels {agree_ * 100:.0f}% equal"
        if bad_ == 0 and err_ < 1e-4 and agree_ == 1.0 and not ckpts:
            st.sidebar.success("PASS: " + msg_)
        else:
            st.sidebar.warning("CHECK: " + msg_)

# ------------------------------------------------------------------------------------------- header
st.markdown(
    "<div style='text-align:center;margin-bottom:18px'>"
    "<div style='font-size:40px;line-height:1'>❤️</div>"
    "<div style='font-size:32px;font-weight:700;color:#1B1F2A;margin-top:6px'>ECG MI Screening</div>"
    f"<div style='color:{MUTED};font-size:16px;margin-top:4px'>Check a 12-lead ECG for signs of a heart attack "
    "in the inferior or anteroseptal wall.</div></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------- step 1: choose an ECG
sig, fs, names, source, truth = None, None, None, None, None
fs_note, ref, upload_recs = None, None, []
with st.container(border=True):
    step(1, "Choose an ECG")
    sources = (["Examples"] if examples else []) + ["Upload"] + (["PTB-XL"] if Path(PTBXL_DEFAULT).is_dir() else [])
    src = choose("Source", sources, sources[0], "src",
                 fmt=lambda x: {"Examples": "Try an example", "Upload": "Upload my ECG", "PTB-XL": "PTB-XL record"}[x])
    if src == "Examples":
        st.markdown("<div class='hint'>Real ECGs from PTB-XL that the model never saw during training.</div>",
                    unsafe_allow_html=True)
        exs = examples["examples"]
        k = choose("Example", list(range(len(exs))), 0, "ex",
                   fmt=lambda i: (f"{NICE[exs[i]['true_class']]} · #{exs[i]['ecg_id']}" if exs[i]["kind"] == "showcase"
                                  else f"Difficult case · #{exs[i]['ecg_id']}"))
        e = exs[k]
        sig, fs, names = pp.read_wfdb(EXAMPLES_DIR / e["record"])
        source, truth = f"PTB-XL #{e['ecg_id']}", e["true_class"]
    elif src == "Upload":
        st.markdown("<div class='hint'>A CSV with one column per lead (I, II, III, aVR, aVL, aVF, V1–V6), "
                    "or WFDB recordings (.hea and .dat together). Several ECGs at once are fine. Any sampling rate. PTB-XL files are checked "
                    "against the cardiologist's label.</div>", unsafe_allow_html=True)
        files = st.file_uploader("12-lead ECG: CSV (lead names in header), WFDB (.hea + .dat) or NPY",
                                 type=["hea", "dat", "mat", "xyz", "csv", "txt", "npy"], accept_multiple_files=True)
        if files:
            upload_recs, problems = read_uploads(files, fs_override)
            for p_ in problems:
                st.warning(p_, icon="📄")
            if len(upload_recs) > 1:
                st.caption(f"{len(upload_recs)} ECGs uploaded.")

    else:
        ecg_id = st.number_input("PTB-XL ecg_id", 1, 21837, 1, 1)
        try:
            row = load_ptbxl_db(PTBXL_DEFAULT).loc[int(ecg_id)]
            sig, fs, names = pp.read_wfdb(Path(PTBXL_DEFAULT) / row["filename_lr"])
            source = f"PTB-XL #{int(ecg_id)} (fold {int(row['strat_fold'])})"
            in_cls = [c_ for c_ in row["scp_codes"] if c_ in CLASS_NAMES]
            truth = in_cls[0] if len(in_cls) == 1 else None
            if int(row["strat_fold"]) <= 8:
                st.caption("This ECG was part of the training data.")
        except Exception as ex:  # noqa: BLE001
            st.error(f"Could not load PTB-XL record: {ex}"); st.stop()

if src == "Upload":
    recs = upload_recs
else:
    recs = [] if sig is None else [dict(sig=sig, fs=fs, names=names, source=source, truth=truth, ref=ref, fs_note=None)]
if not recs:
    st.stop()


def prep(rec):
    s100, nts = pp.prepare_signal(rec["sig"], rec["fs"], rec["names"])
    return s100, ([rec["fs_note"]] if rec.get("fs_note") else []) + nts


batch = len(recs) > 1
if batch:
    prog = st.progress(0.0, text=f"Analysing {len(recs)} ECGs...")
    for i, rec in enumerate(recs):
        try:
            s100, nts = prep(rec)
            rb = analyse(model, model_key, s100, False)
            rec.update(sig100=s100, notes=nts, cls=CLASS_NAMES[rb["label"]], hr=rb["nb"] * 6)
        except Exception as ex:  # noqa: BLE001
            rec.update(error=str(ex))
        prog.progress((i + 1) / len(recs), text=f"Analysing {len(recs)} ECGs...")
    prog.empty()
    ok_recs = [x for x in recs if "error" not in x]
    with st.container(border=True):
        step(2, f"Results for {len(recs)} ECGs")
        labelled = [x for x in ok_recs if x["truth"]]
        agree = sum(x["cls"] == x["truth"] for x in labelled)
        m1, m2, m3 = st.columns(3)
        m1.metric("ECGs analysed", len(ok_recs))
        m2.metric("MI detected", sum(x["cls"] != "NORM" for x in ok_recs))
        m3.metric("Same as cardiologist", f"{agree} of {len(labelled)}" if labelled else "no labels")
        rows_ = []
        for x in recs:
            if "error" in x:
                rows_.append({"ECG": x["source"], "Result": "could not analyse", "Cardiologist": "", "Match": "", "Heart rate": ""})
                continue
            card_ = NICE[x["truth"]] if x["truth"] else (x["ref"]["codes"] if x.get("ref") else "no label")
            match_ = ("✓" if x["cls"] == x["truth"] else "✗") if x["truth"] else "–"
            rows_.append({"ECG": x["source"], "Result": NICE[x["cls"]], "Cardiologist": card_, "Match": match_,
                          "Heart rate": f"~{x['hr']} bpm"})
        st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)
        if not ok_recs:
            st.stop()
        pick = st.selectbox("Show details for", list(range(len(ok_recs))),
                            format_func=lambda i: f"{ok_recs[i]['source']}  →  {NICE[ok_recs[i]['cls']]}")
    rec = ok_recs[pick]
    sig100, notes = rec["sig100"], rec["notes"]
else:
    rec = recs[0]
    try:
        sig100, notes = prep(rec)
    except Exception as ex:  # noqa: BLE001
        st.error(f"This file could not be used: {ex}"); st.stop()
source, truth, ref = rec["source"], rec["truth"], rec["ref"]

with st.spinner("Analysing the ECG..."):
    r = analyse(model, model_key, sig100, explain)
cls = CLASS_NAMES[r["label"]]
imp = r["imp"]
key_leads = [pp.LEADS[i] for i in np.argsort(imp)[::-1][:3]] if imp is not None else []

# ------------------------------------------------------------------------------------------- step 2: result
with st.container(border=True):
    if batch:
        step("▸", f"Details: {source}")
    else:
        step(2, "Result")
    title, icon, text = VERDICT[cls]
    col = GREEN if cls == "NORM" else RED
    chips = f"<span class='chip'>❤ Heart rate ~{r['nb'] * 6} bpm</span><span class='chip'>⏱ {r['seconds']:.1f} s</span>"
    if truth:
        ok = truth == cls
        chips += (f"<span class='chip' style='background:{GREEN if ok else RED}1a;color:{GREEN if ok else RED}'>"
                  f"{'✓ Same as' if ok else '✗ Differs from'} cardiologist ({NICE[truth]})</span>")
    elif ref is not None:
        chips += (f"<span class='chip'>Cardiologist: {ref['codes']} (outside the 3 classes the model was trained on)</span>")
    else:
        chips += "<span class='chip'>No reference label for this file (not a known PTB-XL record)</span>"
    if ref is not None and ref["fold"] <= 8:
        chips += "<span class='chip' style='background:#EDA1001f;color:#8a5a00'>Part of the training data</span>"
    st.markdown(
        f"<div style='background:{col}0f;border:1.5px solid {col}55;border-radius:14px;padding:22px 24px'>"
        f"<div style='font-size:30px;font-weight:700;color:{col}'>{icon}&nbsp; {title}</div>"
        f"<div style='font-size:16px;color:{INK};margin-top:8px;line-height:1.5'>{text}</div>"
        f"<div>{chips}</div></div>", unsafe_allow_html=True)
    if r["flags"]:
        st.warning("Signal quality: " + " ".join(r["flags"]), icon="🔎")
    st.caption(f"{source} · Research prototype, not a medical diagnosis. Always have an ECG reviewed by a doctor.")

# ------------------------------------------------------------------------------------------- step 3: why
with st.container(border=True):
    step(3, "Why?")
    if key_leads:
        sentence = f"The model relied most on leads <b>{', '.join(key_leads)}</b>"
        if cls in EXPECTED:
            hit = len(EXPECTED[cls] & set(key_leads))
            sentence += (", exactly where this type of infarct shows up." if hit >= 2 else
                         ", partly where this type of infarct shows up." if hit == 1 else ".")
        else:
            sentence += "."
        st.markdown(f"<div class='why'>{sentence}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hint'>The typical heartbeat of each lead, drawn on ECG paper"
                + (" (influential leads in red)" if key_leads else "") + ".</div>", unsafe_allow_html=True)
    st.pyplot(plot_heartbeats(r["beat"], key_leads))

    with st.expander("See the full ECG"):
        lead = choose("Lead", pp.LEADS, "II", "lead")
        st.pyplot(plot_strip(r["sig10"], pp.LEADS.index(lead)))

    with st.expander("For experts"):
        if imp is not None:
            st.markdown("**Lead influence:** how much the result's probability drops when that lead is hidden.")
            st.pyplot(plot_importance(imp))
        st.markdown("**What the network sees:** signed beat images (top) and 8–40 Hz spectrograms (bottom).")
        st.pyplot(plot_model_input(r["img"]))
        if notes:
            st.markdown("**Input processing:** " + " ".join(notes))
        if len(r["win_starts"]) > 1:
            st.markdown("**Per 10-second window:** " + ", ".join(
                f"{s / pp.FS:.0f} s → {NICE[CLASS_NAMES[i]]}" for s, i in zip(r["win_starts"], r["win_label"])))
        report = dict(source=source, analysed_at=datetime.now().isoformat(timespec="seconds"),
                      result=NICE[cls], reference_label=truth, beats=int(r["nb"]), quality_flags=r["flags"],
                      preprocessing_notes=notes,
                      lead_influence=None if imp is None else dict(zip(pp.LEADS, np.round(imp, 4).tolist())),
                      disclaimer="Research prototype, not a medical device.")
        st.download_button("Download report (JSON)", json.dumps(report, indent=2),
                           file_name=f"ecg_report_{datetime.now():%Y%m%d_%H%M%S}.json", mime="application/json")

with st.expander("How does it work?"):
    st.markdown(
        "1. **Two pictures per lead:** the typical heartbeat, drawn with its direction (so ST elevation and Q waves "
        "stay visible), and a picture of its fast frequencies (8–40 Hz).\n"
        "2. **A convolutional neural network with two branches** studies the two kinds of pictures, and the branches "
        "exchange information through attention.\n"
        "3. **Three trained networks vote**, and the answer is Normal, Inferior MI or Anteroseptal MI.\n\n"
        "Trained on PTB-XL. On 1,052 validation ECGs: 93% accuracy, macro-AUC 0.96.")
