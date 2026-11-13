"""
ECG MI screening app: NORM / inferior MI (IMI) / anteroseptal MI (ASMI) from a 12-lead ECG.

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
CLASS_LONG = {"NORM": "Normal ECG", "IMI": "Inferior myocardial infarction",
              "ASMI": "Anteroseptal myocardial infarction"}
CLASS_TEXT = {
    "NORM": "No pattern of inferior or anteroseptal infarction was found.",
    "IMI": "Pattern consistent with an inferior MI (leads II, III, aVF).",
    "ASMI": "Pattern consistent with an anteroseptal MI (leads V1-V4).",
}
EXPECTED_LEADS = {"IMI": "II, III, aVF", "ASMI": "V1-V4"}
CLASS_COLOR = {"NORM": "#2E7D5B", "IMI": "#B3122E", "ASMI": "#B3122E"}
INK, MUTED, GRID, RED = "#1B1F2A", "#5A6170", "#E3E5EA", "#B3122E"

st.set_page_config(page_title="ECG MI screening", page_icon="❤️", layout="wide")
st.markdown("""<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem;}
.result-card {padding: 18px 22px; border-radius: 12px;}
.metric-small {color: #5A6170; font-size: 14px;}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


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


# ------------------------------------------------------------------------------------------- plots
def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_ecg(sig10, peaks):
    fig, axes = plt.subplots(4, 3, figsize=(12, 6.2), sharex=True)
    t = np.arange(len(sig10)) / pp.FS
    order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    for k, ax in enumerate(axes.T.flat):
        l = order[k]
        ax.plot(t, sig10[:, l], color=INK, lw=0.7)
        ax.plot(peaks / pp.FS, sig10[peaks, l], "|", color=RED, ms=6, alpha=0.6)
        ax.set_title(pp.LEADS[l], loc="left", fontsize=9, fontweight="bold", pad=2)
        ax.grid(color=GRID, lw=0.5); ax.set_ylim(-2, 2); _clean(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mV")
    fig.tight_layout()
    return fig


def plot_median_beats(beat, importance=None):
    fig, axes = plt.subplots(2, 6, figsize=(12, 3.8), sharey=True)
    t = (np.arange(len(beat)) - pp.PRE) * 1000 / pp.FS
    top = None if importance is None else set(np.argsort(importance)[-3:])
    for l, ax in enumerate(axes.flat):
        hi = top is not None and l in top
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.fill_between(t, 0, beat[:, l], color=RED if hi else "#8A93A6", alpha=0.25)
        ax.plot(t, beat[:, l], color=RED if hi else INK, lw=1.2)
        ax.set_title(pp.LEADS[l] + ("  (key lead)" if hi else ""), fontsize=9,
                     fontweight="bold", color=RED if hi else INK, pad=2)
        ax.set_ylim(-pp.AMP_MV, pp.AMP_MV); ax.grid(color=GRID, lw=0.5)
        ax.set_xticks([-200, 0, 200, 400]); _clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("ms from R peak")
    for ax in axes[:, 0]:
        ax.set_ylabel("mV")
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
    axes[0, 0].set_ylabel("Beat", fontsize=8)
    axes[1, 0].set_ylabel("Spectro.", fontsize=8)
    fig.tight_layout(pad=0.3)
    return fig


def plot_importance(imp, cls_name):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(pp.LEADS, imp, color=[RED if v == imp.max() else "#8A93A6" for v in imp])
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(f"Drop in P({cls_name})")
    ax.grid(axis="y", color=GRID, lw=0.5); _clean(ax)
    fig.tight_layout()
    return fig


def prob_bars(p, title):
    st.markdown(f"**{title}**")
    for k, c in enumerate(CLASS_NAMES):
        st.progress(float(np.clip(p[k], 0, 1)), text=f"{c} - {CLASS_LONG[c]}: {p[k] * 100:.1f}%")


# ------------------------------------------------------------------------------------------- sidebar
examples = load_examples()
PTBXL_DEFAULT = os.environ.get("ECG_MI_PTBXL", "/home/swarnabha/physionet.org/files/ptb-xl/1.0.3")
modes = ["Upload an ECG"] + (["Example ECGs"] if examples else []) + (["PTB-XL record"] if Path(PTBXL_DEFAULT).is_dir() else [])
st.sidebar.header("Input")
mode = st.sidebar.radio("ECG source", modes, index=1 if examples else 0)
explain = st.sidebar.checkbox("Show which leads drove the decision", value=True,
                              help="Occlusion analysis: 12 extra passes through the networks.")

default_project = Path(os.environ.get("ECG_MI_PROJECT", APP_DIR.parent))
with st.sidebar.expander("Model settings", expanded=False):
    project = Path(st.text_input("Project folder (contains runs/)", value=str(default_project)))
    model_src = st.radio("Trained model", ["Final ensemble (3 networks)", "My own checkpoint file(s)"],
                         help="The ensemble = your 3 best.pt files + class offsets from ensemble_config.json. "
                              "Or give any SSPNet .pt file(s) you trained (best.pt or last.pt).")
    ens_dir = str(project / "runs" / "mi_sspnet_v5_ensemble")
    ckpts = None
    if model_src.startswith("Final"):
        ens_dir = st.text_input("Ensemble folder", value=ens_dir)
    else:
        txt = st.text_area("Checkpoint path(s), one per line",
                           value=str(project / "runs" / "mi_sspnet_v5_s1" / "best.pt"))
        ckpts = tuple(p.strip() for p in txt.splitlines() if p.strip())
    ncpu = os.cpu_count() or 2
    threads = st.slider("CPU threads", 1, max(1, ncpu), min(4, ncpu))
    verify = st.button("Verify model against the notebook",
                       help="Recomputes 20 validation ECGs and compares with the notebook's saved images and "
                            "probabilities (selftest_pack.npz).")

st.title("❤️ ECG myocardial infarction screening")
st.caption("12-lead ECG → normal / inferior MI / anteroseptal MI · ensemble of three dual-branch attention CNNs "
           "trained on PTB-XL · runs fully offline on this laptop")

try:
    model = load_model(ens_dir, str(project), threads, ckpts)
    st.sidebar.success(f"Model ready: {len(model.nets)} network(s), offsets b = {np.round(model.bias, 2).tolist()}")
except Exception as e:  # noqa: BLE001
    where = ", ".join(ckpts) if ckpts else ens_dir
    st.error(f"Could not load the trained model from `{where}`.\n\n{e}\n\n"
             "Point *Project folder* to the folder that contains `runs/`. For the ensemble, Section 6b of the "
             "notebook must have been run (it writes `ensemble_config.json`).")
    st.stop()
model_key = f"{ens_dir}|{ckpts}|{threads}"

if verify:
    pack = APP_DIR / "selftest_pack.npz"
    if not pack.is_file():
        st.sidebar.error("selftest_pack.npz not found (it is created by export_bundle.py).")
    else:
        with st.spinner("Verifying the model on validation ECGs..."):
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
        msg_ = (f"Images identical: {n_ - bad_}/{n_} · max probability difference {err_:.1e} · "
                f"same final label {agree_ * 100:.0f}%")
        if bad_ == 0 and err_ < 1e-4 and agree_ == 1.0 and not ckpts:
            st.sidebar.success("PASS - identical to the notebook. " + msg_)
        else:
            st.sidebar.warning(("CHECK - " if not ckpts else "Own checkpoint (reference is the ensemble): ") + msg_)

# ------------------------------------------------------------------------------------------- input
sig, fs, names, source, truth = None, None, None, None, None
if mode == "Example ECGs":
    exs = examples["examples"]
    fold_name = "test" if examples["fold"] == 10 else "validation"
    labels = [f"{'⭐' if e['kind'] == 'showcase' else '🧩'}  {e['true_class']} · ecg_id {e['ecg_id']}"
              + ("  (hard case)" if e["kind"] != "showcase" else "") for e in exs]
    k = st.sidebar.selectbox(f"PTB-XL {fold_name}-fold ECG (never seen in training)", range(len(exs)),
                             format_func=lambda i: labels[i])
    e = exs[k]
    sig, fs, names = pp.read_wfdb(EXAMPLES_DIR / e["record"])
    who = ", ".join(x for x in [f"{e['age']} y" if e.get("age") is not None else None, e.get("sex")] if x)
    source = f"PTB-XL ecg_id {e['ecg_id']} ({fold_name} fold{', ' + who if who else ''})"
    truth = e["true_class"]
    up = examples.get("upload_example")
    if up:
        st.sidebar.caption(f"For the upload demo use `demo_examples/{up['file']}` "
                           f"(ECG-machine style: {up['sampling_rate']} Hz, microvolts, shuffled leads).")
elif mode == "Upload an ECG":
    st.markdown("#### 1 · Upload a 12-lead ECG signal")
    u1, u2 = st.columns([3, 1])
    files = u1.file_uploader("WFDB (.hea + .dat together), CSV with lead names in the header, or NPY (samples × 12)",
                             type=["hea", "dat", "csv", "txt", "npy"], accept_multiple_files=True)
    fs_user = u2.number_input("Sampling rate of CSV / NPY (Hz)", 50.0, 2000.0, 500.0, 50.0,
                              help="WFDB files carry their own sampling rate.")
    if files:
        by_ext = {Path(f.name).suffix.lower(): f for f in files}
        try:
            if ".hea" in by_ext:
                if ".dat" not in by_ext:
                    st.error("WFDB needs both the .hea and the .dat file."); st.stop()
                with tempfile.TemporaryDirectory() as td:
                    for f in files:
                        (Path(td) / f.name).write_bytes(f.getvalue())
                    sig, fs, names = pp.read_wfdb(Path(td) / Path(by_ext[".hea"].name).stem)
                source = by_ext[".hea"].name
            elif ".csv" in by_ext or ".txt" in by_ext:
                f = by_ext.get(".csv") or by_ext[".txt"]
                sig, names = pp.read_table(f.getvalue(), f.name)
                fs, source = fs_user, f.name
            elif ".npy" in by_ext:
                f = by_ext[".npy"]
                sig = np.load(f, allow_pickle=False)
                fs, names, source = fs_user, None, f.name
            else:
                st.error("Unsupported file type."); st.stop()
        except Exception as ex:  # noqa: BLE001
            st.error(f"Could not read the file: {ex}"); st.stop()
        if examples and examples.get("upload_example") and Path(source).name == examples["upload_example"]["file"]:
            truth = examples["upload_example"]["true_class"]
else:
    root = st.sidebar.text_input("PTB-XL folder", value=PTBXL_DEFAULT)
    ecg_id = st.sidebar.number_input("ecg_id", 1, 21837, 1, 1)
    try:
        row = load_ptbxl_db(root).loc[int(ecg_id)]
        sig, fs, names = pp.read_wfdb(Path(root) / row["filename_lr"])
        source = f"PTB-XL ecg_id {int(ecg_id)} (fold {int(row['strat_fold'])})"
        codes = row["scp_codes"]
        st.sidebar.caption("PTB-XL statements: " + ", ".join(f"{c} ({v:.0f}%)" for c, v in codes.items()))
        in_cls = [c for c in codes if c in CLASS_NAMES]
        truth = in_cls[0] if len(in_cls) == 1 else None
        if int(row["strat_fold"]) <= 8:
            st.sidebar.warning("Training fold: the model has seen this ECG.")
    except Exception as ex:  # noqa: BLE001
        st.error(f"Could not load PTB-XL record: {ex}"); st.stop()

# ------------------------------------------------------------------------------------------- how it works
main_tab, how_tab = st.tabs(["🩺 Analysis", "⚙️ How it works"])
with how_tab:
    c = st.columns(4)
    steps = [("1 · Two images per lead",
              "The ECG is filtered (0.5-40 Hz), beats are found, and the **median heartbeat** of each lead is drawn "
              "on a fixed ±1.5 mV scale, so ST elevation/depression and Q waves keep their sign. "
              "A **8-40 Hz spectrogram** of each lead adds the frequency content over 10 s."),
             ("2 · Dual-branch CNN",
              "Each image type goes through its own branch of **asymmetric convolutions** (3×3 + 3×1 or 1×3) "
              "with **channel and spatial attention**, giving 16×16 = 256 feature tokens."),
             ("3 · Cross-attention fusion",
              "The two branches **attend to each other**; their sum and difference are fused, refined by "
              "self-attention and turned into probabilities for NORM, IMI and ASMI."),
             ("4 · Ensemble + decision",
              "**Three networks** trained with different seeds are averaged. A class offset tuned on validation "
              "data (IMI +0.5) counters the large NORM majority.")]
    for col, (h, t) in zip(c, steps):
        col.markdown(f"#### {h}\n{t}")
    st.divider()
    m = st.columns(4)
    m[0].metric("Validation macro-AUC", "0.961")
    m[1].metric("Validation accuracy", "93.4%")
    m[2].metric("ASMI recall", "91%")
    m[3].metric("IMI recall", "60%")
    st.caption("PTB-XL validation fold 9 (1,052 ECGs). Inferior MI is the hardest class: many are old infarcts "
               "with only small Q waves.")
    st.warning("**Research prototype, not a medical device.** The model only knows NORM, IMI and ASMI; ECGs with "
               "other findings are still forced into one of these classes.", icon="⚠️")

with main_tab:
    if sig is None:
        st.info("Upload a 12-lead ECG above, or choose *Example ECGs* / *PTB-XL record* in the sidebar.")
        st.stop()
    try:
        sig100, notes = pp.prepare_signal(sig, fs, names)
    except Exception as ex:  # noqa: BLE001
        st.error(str(ex)); st.stop()

    with st.spinner(f"Analysing: building images, running {len(model.nets)} network(s)" + (", measuring lead importance" if explain else "")):
        r = analyse(model, model_key, sig100, explain)
    cls = CLASS_NAMES[r["label"]]
    nwin = len(r["win_starts"])

    st.subheader(source)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(
            f"<div class='result-card' style='background:{CLASS_COLOR[cls]}14;border:1px solid {CLASS_COLOR[cls]}55'>"
            f"<div class='metric-small'>Model output</div>"
            f"<div style='font-size:36px;font-weight:700;color:{CLASS_COLOR[cls]}'>{cls}</div>"
            f"<div style='font-size:18px;font-weight:600;color:{INK}'>{CLASS_LONG[cls]}</div>"
            f"<div style='margin-top:6px;color:{INK}'>{CLASS_TEXT[cls]}</div></div>",
            unsafe_allow_html=True)
        if truth:
            if truth == cls:
                st.success(f"Cardiologist label: **{truth}** — the model agrees ✓")
            else:
                st.error(f"Cardiologist label: **{truth}** — the model says {cls} ✗")
        votes = [CLASS_NAMES[int(np.argmax(m_))] for m_ in r["member"]]
        st.caption(f"Network votes: {' · '.join(votes)}  |  {nwin} × 10-s window(s)  |  "
                   f"{r['nb']} beats (~{r['nb'] * 6} bpm)  |  analysed in {r['seconds']:.1f} s on CPU")
    with c2:
        prob_bars(r["p_adj"], "Probabilities (after the decision rule)")
        with st.expander("Per-network probabilities"):
            tab = pd.DataFrame(r["member"], columns=CLASS_NAMES,
                               index=[f"Network {i + 1}" for i in range(len(r["member"]))])
            tab.loc["Ensemble mean"] = r["p_rec"]
            tab.loc["After offsets"] = r["p_adj"]
            st.dataframe(tab.style.format("{:.3f}"), use_container_width=True)

    for n in notes:
        st.caption("ℹ️ " + n)
    for f in r["flags"]:
        st.warning(f, icon="🔎")
    if nwin > 1:
        wt = pd.DataFrame(r["win_prob"], columns=CLASS_NAMES)
        wt.insert(0, "start (s)", [s / pp.FS for s in r["win_starts"]])
        wt.insert(1, "prediction", [CLASS_NAMES[i] for i in r["win_label"]])
        st.dataframe(wt.style.format({c_: "{:.3f}" for c_ in CLASS_NAMES}), use_container_width=True, hide_index=True)

    imp = r["imp"]
    t1, t2, t3, t4 = st.tabs(["Median beats", "Lead importance", "12-lead ECG", "Model input images"])
    with t1:
        st.caption("Median heartbeat of every lead, baseline = PR segment; shaded area = deviation from 0 mV "
                   "(what the beat branch sees)." + (" Red = the three most influential leads." if imp is not None else ""))
        st.pyplot(plot_median_beats(r["beat"], imp))
    with t2:
        if imp is None:
            st.info("Enable *Show which leads drove the decision* in the sidebar.")
        else:
            top3 = ", ".join(pp.LEADS[i] for i in np.argsort(imp)[::-1][:3])
            exp = f" Expected for {cls}: {EXPECTED_LEADS[cls]}." if cls in EXPECTED_LEADS else ""
            st.caption(f"How much P({cls}) drops when one lead is hidden from the model. Top leads: **{top3}**.{exp}")
            st.pyplot(plot_importance(imp, cls))
    with t3:
        st.caption(f"10-s window starting at {r['win_starts'][r['best_w']] / pp.FS:.0f} s, resampled to 100 Hz. "
                   "Red ticks = detected R peaks.")
        st.pyplot(plot_ecg(r["sig10"], r["peaks"]))
    with t4:
        st.caption("The 2 × 12 images of 128 × 128 pixels given to the network: signed median beats (top) and "
                   "8-40 Hz spectrograms (bottom).")
        st.pyplot(plot_model_input(r["img"]))

    report = dict(
        source=source, analysed_at=datetime.now().isoformat(timespec="seconds"),
        prediction=cls, description=CLASS_LONG[cls], reference_label=truth,
        probabilities_after_decision_rule=dict(zip(CLASS_NAMES, np.round(r["p_adj"], 4).tolist())),
        ensemble_mean_probabilities=dict(zip(CLASS_NAMES, np.round(r["p_rec"], 4).tolist())),
        decision_offsets=np.round(model.bias, 3).tolist(), n_windows=nwin, beats=int(r["nb"]),
        quality_flags=r["flags"], preprocessing_notes=notes,
        lead_importance=None if imp is None else dict(zip(pp.LEADS, np.round(imp, 4).tolist())),
        disclaimer="Research prototype, not a medical device.")
    st.download_button("⬇️ Download report (JSON)", json.dumps(report, indent=2),
                       file_name=f"ecg_mi_report_{datetime.now():%Y%m%d_%H%M%S}.json", mime="application/json")
    st.caption("⚠️ Research prototype - not a medical device. Not for clinical decisions.")
