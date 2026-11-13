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
CLASS_LONG = {"NORM": "Normal", "IMI": "Inferior MI", "ASMI": "Anteroseptal MI"}
CLASS_TEXT = {
    "NORM": "No sign of inferior or anteroseptal infarction.",
    "IMI": "Infarct pattern in the inferior leads (II, III, aVF).",
    "ASMI": "Infarct pattern in the anteroseptal leads (V1-V4).",
}
EXPECTED_LEADS = {"IMI": "II, III, aVF", "ASMI": "V1-V4"}
CLASS_COLOR = {"NORM": "#2E7D5B", "IMI": "#B3122E", "ASMI": "#B3122E"}
INK, MUTED, GRID, RED = "#1B1F2A", "#5A6170", "#E3E5EA", "#B3122E"

st.set_page_config(page_title="ECG MI Screening", page_icon="❤️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-size: 26px;}
.stTabs [data-baseweb="tab"] {font-size: 15px;}
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
        ax.set_title(pp.LEADS[l], fontsize=10,
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
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.bar(pp.LEADS, imp, color=[RED if v == imp.max() else "#8A93A6" for v in imp])
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(f"Drop in P({cls_name})")
    ax.grid(axis="y", color=GRID, lw=0.5); _clean(ax)
    fig.tight_layout()
    return fig


def plot_one_lead(sig10, peaks, lead):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    t = np.arange(len(sig10)) / pp.FS
    ax.plot(t, sig10[:, lead], color=INK, lw=0.9)
    ax.plot(peaks / pp.FS, sig10[peaks, lead], "o", color=RED, ms=4)
    ax.set(xlabel="Time (s)", ylabel="mV", ylim=(-2, 2)); ax.grid(color=GRID, lw=0.5); _clean(ax)
    fig.tight_layout()
    return fig


# ---- interactive charts (Altair ships with Streamlit); each falls back to a static image if anything fails
def chart_importance(imp, cls_name):
    try:
        import altair as alt
        top = set(np.argsort(imp)[-3:])
        df = pd.DataFrame({"lead": pp.LEADS, "drop": np.round(imp, 4), "key": [i in top for i in range(12)]})
        ch = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
              .encode(x=alt.X("lead:N", sort=pp.LEADS, title=None, axis=alt.Axis(labelAngle=0)),
                      y=alt.Y("drop:Q", title=f"Drop in P({cls_name})"),
                      color=alt.condition(alt.datum.key, alt.value(RED), alt.value("#8A93A6")),
                      tooltip=[alt.Tooltip("lead:N", title="Lead"), alt.Tooltip("drop:Q", title="Probability drop", format=".3f")])
              .properties(height=260))
        st.altair_chart(ch, use_container_width=True)
    except Exception:  # noqa: BLE001
        st.pyplot(plot_importance(imp, cls_name))


def chart_lead(sig10, peaks, lead):
    try:
        import altair as alt
        t = np.arange(len(sig10)) / pp.FS
        df = pd.DataFrame({"t": np.round(t, 2), "mV": np.round(sig10[:, lead], 3)})
        pk = pd.DataFrame({"t": np.round(peaks / pp.FS, 2), "mV": np.round(sig10[peaks, lead], 3)})
        base = alt.Chart(df).mark_line(color=INK, strokeWidth=1.2).encode(
            x=alt.X("t:Q", title="Time (s)"), y=alt.Y("mV:Q", title="mV", scale=alt.Scale(domain=[-2, 2])),
            tooltip=[alt.Tooltip("t:Q", title="s"), alt.Tooltip("mV:Q", title="mV")])
        dots = alt.Chart(pk).mark_circle(color=RED, size=45).encode(x="t:Q", y="mV:Q")
        st.altair_chart((base + dots).properties(height=280).interactive(bind_y=False), use_container_width=True)
    except Exception:  # noqa: BLE001
        st.pyplot(plot_one_lead(sig10, peaks, lead))


def choose(label, options, default, key, fmt=None):
    """Clickable chips if this Streamlit has them, otherwise horizontal radio buttons."""
    fmt = fmt or (lambda x: x)
    if hasattr(st, "pills"):
        v = st.pills(label, options, default=default, format_func=fmt, key=key, label_visibility="collapsed")
        return default if v is None else v
    return st.radio(label, options, index=options.index(default), format_func=fmt, key=key,
                    horizontal=True, label_visibility="collapsed")


def result_card(cls, truth):
    col = CLASS_COLOR[cls]
    chip = ""
    if truth:
        ok = truth == cls
        chip = (f"<span style='display:inline-block;margin-top:10px;padding:3px 10px;border-radius:999px;"
                f"font-size:13px;background:{'#2E7D5B' if ok else '#B3122E'}1f;color:{'#2E7D5B' if ok else '#B3122E'}'>"
                f"{'✓ matches' if ok else '✗ differs from'} cardiologist label ({truth})</span>")
    st.markdown(
        f"<div style='padding:20px 24px;border-radius:14px;background:{col}12;border:1px solid {col}40'>"
        f"<div style='font-size:13px;color:{MUTED};letter-spacing:.04em'>RESULT</div>"
        f"<div style='font-size:44px;font-weight:700;line-height:1.1;color:{col}'>{CLASS_LONG[cls]}</div>"
        f"<div style='font-size:15px;color:{INK};margin-top:4px'>{CLASS_TEXT[cls]}</div>{chip}</div>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- settings (sidebar)
examples = load_examples()
PTBXL_DEFAULT = os.environ.get("ECG_MI_PTBXL", "/home/swarnabha/physionet.org/files/ptb-xl/1.0.3")
default_project = Path(os.environ.get("ECG_MI_PROJECT", APP_DIR.parent))
st.sidebar.markdown("### ⚙️ Settings")
explain = st.sidebar.toggle("Lead importance", value=True, help="Shows which leads drove the decision (a little slower).") \
    if hasattr(st.sidebar, "toggle") else st.sidebar.checkbox("Lead importance", value=True)
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
    ncpu = os.cpu_count() or 2
    threads = st.slider("CPU threads", 1, max(1, ncpu), min(4, ncpu))
    verify = st.button("Verify model", help="Recomputes 20 validation ECGs and compares with the training notebook.")

try:
    model = load_model(ens_dir, str(project), threads, ckpts)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the trained model.\n\n{e}\n\nCheck *Settings → Model → Project folder* (it must contain `runs/`).")
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
    "<div style='margin-bottom:4px'><span style='font-size:34px;font-weight:700;color:#1B1F2A'>❤️ ECG MI Screening</span></div>"
    f"<div style='color:{MUTED};font-size:16px;margin-bottom:6px'>Normal · Inferior MI · Anteroseptal MI from a 12-lead ECG</div>",
    unsafe_allow_html=True)
main_tab, how_tab = st.tabs(["🩺 Analyse", "⚙️ How it works"])

with how_tab:
    c = st.columns(4)
    steps = [("1 · Two images per lead", "Each lead's **median heartbeat**, drawn with its sign (ST up or down), plus its **8-40 Hz spectrogram**."),
             ("2 · Dual-branch CNN", "Each image type has its own branch: **asymmetric convolutions** + **attention**."),
             ("3 · Cross-attention", "The two branches **look at each other** and are fused into one decision."),
             ("4 · Ensemble", "**Three networks** are averaged; a small offset favours MI over the NORM majority.")]
    for col, (h, t) in zip(c, steps):
        col.markdown(f"**{h}**\n\n{t}")
    st.divider()
    m = st.columns(4)
    m[0].metric("Macro-AUC", "0.961")
    m[1].metric("Accuracy", "93.4%")
    m[2].metric("ASMI recall", "91%")
    m[3].metric("IMI recall", "60%")
    st.caption("Validation fold of PTB-XL (1,052 ECGs).")

# ------------------------------------------------------------------------------------------- input
with main_tab:
    sources = (["Examples"] if examples else []) + ["Upload"] + (["PTB-XL"] if Path(PTBXL_DEFAULT).is_dir() else [])
    src = choose("Source", sources, sources[0], "src",
                 fmt=lambda x: {"Examples": "📋 Examples", "Upload": "⬆️ Upload ECG", "PTB-XL": "🗂️ PTB-XL record"}[x])

    sig, fs, names, source, truth = None, None, None, None, None
    if src == "Examples":
        exs = examples["examples"]
        lab = {"NORM": "Normal", "IMI": "Inferior MI", "ASMI": "Anteroseptal MI"}
        opts = list(range(len(exs)))
        k = choose("Example", opts, 0, "ex",
                   fmt=lambda i: (f"{lab[exs[i]['true_class']]} · {exs[i]['ecg_id']}" if exs[i]["kind"] == "showcase"
                                  else f"🧩 Hard case · {exs[i]['ecg_id']}"))
        e = exs[k]
        sig, fs, names = pp.read_wfdb(EXAMPLES_DIR / e["record"])
        source, truth = f"PTB-XL #{e['ecg_id']}", e["true_class"]
    elif src == "Upload":
        u1, u2 = st.columns([5, 1])
        files = u1.file_uploader("12-lead ECG: CSV (lead names in header), WFDB (.hea + .dat) or NPY",
                                 type=["hea", "dat", "csv", "txt", "npy"], accept_multiple_files=True)
        fs_user = u2.number_input("Hz", 50.0, 2000.0, 500.0, 50.0, help="Sampling rate of a CSV / NPY file.")
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
            except Exception as ex:  # noqa: BLE001
                st.error(f"Could not read the file: {ex}"); st.stop()
            if examples and examples.get("upload_example") and Path(source).name == examples["upload_example"]["file"]:
                truth = examples["upload_example"]["true_class"]
    else:
        ecg_id = st.number_input("PTB-XL ecg_id", 1, 21837, 1, 1)
        try:
            row = load_ptbxl_db(PTBXL_DEFAULT).loc[int(ecg_id)]
            sig, fs, names = pp.read_wfdb(Path(PTBXL_DEFAULT) / row["filename_lr"])
            source = f"PTB-XL #{int(ecg_id)} (fold {int(row['strat_fold'])})"
            in_cls = [c_ for c_ in row["scp_codes"] if c_ in CLASS_NAMES]
            truth = in_cls[0] if len(in_cls) == 1 else None
            if int(row["strat_fold"]) <= 8:
                st.caption("⚠️ Training fold: the model has seen this ECG.")
        except Exception as ex:  # noqa: BLE001
            st.error(f"Could not load PTB-XL record: {ex}"); st.stop()

    if sig is None:
        st.info("Drop a 12-lead ECG file above to analyse it.")
        st.stop()
    try:
        sig100, notes = pp.prepare_signal(sig, fs, names)
    except Exception as ex:  # noqa: BLE001
        st.error(str(ex)); st.stop()

    with st.spinner("Analysing..."):
        r = analyse(model, model_key, sig100, explain)
    cls = CLASS_NAMES[r["label"]]
    nwin = len(r["win_starts"])

    # ---------------------------------------------------------------------------------------- result
    st.write("")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        result_card(cls, truth)
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Heart rate", f"~{r['nb'] * 6} bpm")
        m2.metric("Analysed in", f"{r['seconds']:.1f} s")
        st.caption(f"{source} · {nwin} × 10 s")
    for f in r["flags"]:
        st.warning(f, icon="🔎")
    if notes or nwin > 1:
        with st.expander("Input details"):
            for n in notes:
                st.write("• " + n)
            if nwin > 1:
                st.dataframe(pd.DataFrame({"start (s)": [s / pp.FS for s in r["win_starts"]],
                                           "result": [CLASS_NAMES[i] for i in r["win_label"]]}),
                             use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------------------------- evidence
    imp = r["imp"]
    t1, t2, t3, t4 = st.tabs(["💓 Heartbeats", "🎯 Key leads", "📈 ECG", "🧠 Model input"])
    with t1:
        st.caption("Typical heartbeat of each lead" + (" · red = most influential leads" if imp is not None else ""))
        st.pyplot(plot_median_beats(r["beat"], imp))
    with t2:
        if imp is None:
            st.info("Turn on *Lead importance* in Settings.")
        else:
            top3 = [pp.LEADS[i] for i in np.argsort(imp)[::-1][:3]]
            line = f"Top leads: **{', '.join(top3)}**"
            if cls in EXPECTED_LEADS:
                expected = {"IMI": {"II", "III", "aVF"}, "ASMI": {"V1", "V2", "V3", "V4"}}[cls]
                hit = len(expected & set(top3))
                line += f" · expected for {cls}: {EXPECTED_LEADS[cls]} " + ("✅" if hit >= 2 else ("☑️" if hit == 1 else "⚠️"))
            st.markdown(line)
            chart_importance(imp, cls)
            st.caption("Bar = how much the result's probability drops when that lead is hidden. Hover for values.")
    with t3:
        lead_opts = pp.LEADS + ["All 12"]
        sel = choose("Lead", lead_opts, "II", "lead")
        if sel == "All 12":
            st.pyplot(plot_ecg(r["sig10"], r["peaks"]))
        else:
            chart_lead(r["sig10"], r["peaks"], pp.LEADS.index(sel))
            st.caption("Red dots = detected heartbeats · scroll to zoom, drag to pan, double-click to reset.")
    with t4:
        st.caption("What the network sees: signed beat images (top) and spectrograms (bottom).")
        st.pyplot(plot_model_input(r["img"]))

    report = dict(source=source, analysed_at=datetime.now().isoformat(timespec="seconds"),
                  prediction=cls, description=CLASS_LONG[cls], reference_label=truth,
                  n_windows=nwin, beats=int(r["nb"]), quality_flags=r["flags"], preprocessing_notes=notes,
                  lead_importance=None if imp is None else dict(zip(pp.LEADS, np.round(imp, 4).tolist())),
                  disclaimer="Research prototype, not a medical device.")
    d1, d2 = st.columns([1, 3])
    d1.download_button("⬇️ Report", json.dumps(report, indent=2),
                       file_name=f"ecg_mi_report_{datetime.now():%Y%m%d_%H%M%S}.json", mime="application/json")
    d2.caption("Research prototype · not a medical device")
