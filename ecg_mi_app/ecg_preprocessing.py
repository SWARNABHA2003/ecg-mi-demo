"""
ECG loading and preprocessing for the MI app.

The image-making functions are copied unchanged from the training notebook, so the app
feeds the network exactly what it was trained on:
  channel 0 (alpha branch): signed median-beat trace image per lead
  channel 1 (delta branch): 8-40 Hz STFT magnitude image per lead
No PyTorch needed in this file.
"""
from pathlib import Path
import io

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import stft, butter, sosfiltfilt, find_peaks, resample_poly
from fractions import Fraction

FS = 100
WIN = 10 * FS                       # the network was trained on 10-second recordings
NPERSEG, NOVERLAP, NFFT = 128, 120, 512
BANDS = [(0.5, 8.0), (8.0, 40.0)]
IMG = 128
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# =============================================================================================
# Copied from the training notebook (do not change, or the app no longer matches the model)
# =============================================================================================
def ecg_to_12lead_portraits(sig, fs=FS):
    """(1000,12) mV ECG -> uint8 (2,12,128,128), 12 leads kept separate."""
    sig = np.asarray(sig)
    if sig.ndim != 2 or sig.shape[1] != 12 or sig.shape[0] < NPERSEG:
        raise ValueError(f'Expected ECG array (samples >= {NPERSEG}, 12), got {sig.shape}')
    f, _, Z = stft(sig.T, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
    mag = np.log1p(np.abs(Z))  # (12, frequencies, frames)
    out = np.empty((2, 12, IMG, IMG), dtype=np.uint8)
    for b, (lo, hi) in enumerate(BANDS):
        rows = np.where((f >= lo) & (f < hi))[0]
        if len(rows) == 0:
            raise ValueError(f'No STFT bins for band {lo}-{hi} Hz')
        band = mag[:, rows, :][:, ::-1, :]
        resized = zoom(band, (1.0, IMG / band.shape[1], IMG / band.shape[2]), order=1)
        assert resized.shape == (12, IMG, IMG)
        lo_c = resized.min(axis=(1, 2), keepdims=True)
        hi_c = resized.max(axis=(1, 2), keepdims=True)
        out[b] = np.clip(255.0 * (resized - lo_c) / (hi_c - lo_c + 1e-12), 0, 255).astype(np.uint8)
    return out


PRE, POST = 30, 60
BEAT_LEN = PRE + POST
AMP_MV = 1.5
_SOS_ECG = butter(3, [0.5, 40.0], btype='bandpass', fs=FS, output='sos')
_SOS_QRS = butter(2, [5.0, 18.0], btype='bandpass', fs=FS, output='sos')


def detect_r_peaks(sig_f, fs=FS):
    qrs = sosfiltfilt(_SOS_QRS, sig_f, axis=0)
    env = np.sqrt((qrs ** 2).sum(1))
    k = max(1, int(0.08 * fs))
    env = np.convolve(env, np.ones(k) / k, mode='same')
    top = np.percentile(env, 99)
    if not np.isfinite(top) or top <= 1e-6:
        return np.array([], dtype=int), env
    peaks, _ = find_peaks(env, height=0.35 * top, distance=int(0.30 * fs))
    return peaks, env


def median_beat(sig_f, peaks, env):
    n = len(sig_f)
    ok = [p for p in peaks if p - PRE >= 0 and p + POST <= n]
    if ok:
        beats = np.stack([sig_f[p - PRE:p + POST] for p in ok])
        beat = np.median(beats, axis=0)
    else:
        c = int(np.clip(np.argmax(env), PRE, n - POST))
        beat = sig_f[c - PRE:c + POST].copy()
    beat = beat - beat[PRE - 12:PRE - 4].mean(0, keepdims=True)
    return beat, len(ok)


def beat_to_trace_images(beat):
    out = np.zeros((12, IMG, IMG), dtype=np.uint8)
    t_src, t_dst = np.arange(len(beat)), np.linspace(0, len(beat) - 1, IMG)
    zero = int(round((IMG - 1) / 2))
    for l in range(12):
        v = np.interp(t_dst, t_src, beat[:, l])
        r = np.clip(np.rint((AMP_MV - v) / (2 * AMP_MV) * (IMG - 1)), 0, IMG - 1).astype(int)
        for c in range(IMG):
            lo, hi = sorted((r[c], zero))
            out[l, lo:hi + 1, c] = 96
        for c in range(IMG):
            a = r[c] if c == 0 else r[c - 1]
            lo, hi = sorted((a, r[c]))
            out[l, max(lo - 1, 0):min(hi + 1, IMG - 1) + 1, c] = 255
    return out


def ecg_to_v4_portraits(sig, fs=FS):
    """(1000, 12) mV ECG -> (uint8 (2, 12, 128, 128), n_beats)."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != 12 or sig.shape[0] < NPERSEG:
        raise ValueError(f'Expected ECG array (samples >= {NPERSEG}, 12), got {sig.shape}')
    sig_f = sosfiltfilt(_SOS_ECG, sig, axis=0)
    peaks, env = detect_r_peaks(sig_f, fs)
    beat, nb = median_beat(sig_f, peaks, env)
    out = np.empty((2, 12, IMG, IMG), dtype=np.uint8)
    out[0] = beat_to_trace_images(beat)
    out[1] = ecg_to_12lead_portraits(sig, fs)[1]
    return out, nb

# =============================================================================================
# App additions: reading files, unit / rate handling, windows and quality checks
# =============================================================================================
_ALIASES = {'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF', 'D1': 'I', 'D2': 'II', 'D3': 'III',
            'LEAD I': 'I', 'LEAD II': 'II', 'LEAD III': 'III'}


def canonical_lead(name):
    s = str(name).strip()
    u = s.upper().replace('LEAD_', 'LEAD ').strip()
    if u in _ALIASES:
        return _ALIASES[u]
    for l in LEADS:
        if u == l.upper():
            return l
    return s


def reorder_leads(sig, names):
    """Put columns into I, II, III, aVR, aVL, aVF, V1..V6 order using their names."""
    names = [canonical_lead(n) for n in names]
    missing = [l for l in LEADS if l not in names]
    if missing:
        raise ValueError(f'Missing leads {missing}. Found: {names}. All 12 standard leads are required.')
    return sig[:, [names.index(l) for l in LEADS]]


def resample_to_fs(sig, fs_in, fs_out=FS):
    if int(round(fs_in)) == fs_out:
        return sig
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(sig, fr.numerator, fr.denominator, axis=0)


def read_wfdb(record_path):
    """record_path without extension (needs <name>.hea and <name>.dat next to each other)."""
    import wfdb
    sig, info = wfdb.rdsamp(str(record_path))
    units = [str(u).lower() for u in (info.get('units') or [])]
    sig = np.asarray(sig, dtype=np.float64)
    if units and all(u in ('uv', 'μv', 'microvolt') for u in units):
        sig = sig / 1000.0
    return sig, float(info['fs']), list(info['sig_name'])


def read_table(file_bytes, filename):
    """CSV / TXT with a header row of lead names (12 columns, optional time column)."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
    drop = [c for c in df.columns if str(c).strip().lower() in ('time', 't', 'sample', 'index', 'unnamed: 0')]
    df = df.drop(columns=drop)
    return df.to_numpy(dtype=np.float64), [str(c) for c in df.columns]


def guess_units_to_mv(sig):
    """PTB-XL is in millivolts. Very large amplitudes usually mean microvolts."""
    p99 = np.nanpercentile(np.abs(sig), 99)
    if p99 > 50:
        return sig / 1000.0, 'Amplitudes look like microvolts; converted to millivolts.'
    return sig, None


def prepare_signal(sig, fs, names):
    """Any 12-lead recording -> float (n, 12) in mV at 100 Hz, standard lead order, plus notes."""
    notes = []
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError(f'Expected a 2-D array (samples x leads), got shape {sig.shape}')
    if sig.shape[0] == 12 and sig.shape[1] != 12:
        sig = sig.T
    if names is not None:
        sig = reorder_leads(sig, names)
    elif sig.shape[1] != 12:
        raise ValueError(f'Expected 12 leads, got {sig.shape[1]}')
    else:
        notes.append('No lead names given: assumed order I, II, III, aVR, aVL, aVF, V1-V6.')
    bad = ~np.isfinite(sig)
    if bad.any():
        notes.append(f'{int(bad.sum())} missing / infinite samples set to 0.')
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig, unit_note = guess_units_to_mv(sig)
    if unit_note:
        notes.append(unit_note)
    if int(round(fs)) != FS:
        notes.append(f'Resampled from {fs:g} Hz to {FS} Hz (the model was trained at {FS} Hz).')
        sig = resample_to_fs(sig, fs)
    if len(sig) < WIN:
        notes.append(f'Recording is {len(sig) / FS:.1f} s; padded to 10 s by repeating it (the model expects 10 s).')
        reps = int(np.ceil(WIN / len(sig)))
        sig = np.tile(sig, (reps, 1))[:WIN]
    return sig, notes


def windows_10s(sig):
    """Non-overlapping 10-s windows (a trailing part >= 5 s gets its own window aligned to the end)."""
    n = len(sig)
    starts = list(range(0, n - WIN + 1, WIN))
    if n - (starts[-1] + WIN) >= WIN // 2:
        starts.append(n - WIN)
    return [(s, sig[s:s + WIN]) for s in starts]


def quality_flags(sig10, n_beats):
    """Simple checks that tell the user when the result should not be trusted."""
    flags = []
    hr = n_beats * 6
    if n_beats < 4:
        flags.append(f'Only {n_beats} complete beats detected: the median beat may be unreliable.')
    elif hr > 150:
        flags.append(f'Detected about {hr} beats/min: possible tachycardia or noise; median beat may be distorted.')
    std = sig10.std(0)
    flat = [LEADS[i] for i in np.where(std < 0.01)[0]]
    if flat:
        flags.append(f'Flat / disconnected lead(s): {", ".join(flat)}.')
    sat = [LEADS[i] for i in np.where(np.abs(sig10).max(0) > 5)[0]]
    if sat:
        flags.append(f'Very large amplitude (> 5 mV) in: {", ".join(sat)}; check units or electrode contact.')
    return flags


def median_beat_mv(sig10):
    """Median beat in mV (for plotting), same steps as the alpha image."""
    sig_f = sosfiltfilt(_SOS_ECG, np.asarray(sig10, dtype=np.float64), axis=0)
    peaks, env = detect_r_peaks(sig_f)
    beat, nb = median_beat(sig_f, peaks, env)
    return beat, peaks, nb
