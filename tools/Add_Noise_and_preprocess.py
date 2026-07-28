"""
Unified Stationary Earth-Rate & Local PSD Noise Pipeline

This script combines:
1. CSV-to-NPY Windowing & Resampling (Replaces csv2npy.py)
2. Train/Val/Test session splitting
3. Local PSD-Based Synthetic Noise Generation (FDM)
4. Diagnostic Evaluation Plotting (>3s tracking, annotated with Session/Angle)
"""

import glob
import math
import os
import re
import zlib
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Iterable

import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

# Attempt to load allantools for ADEV calculation
try:
    import allantools
    _HAVE_ALLANTOOLS = True
except ImportError:
    _HAVE_ALLANTOOLS = False


# =========================================================================
# 1. DATACLASSES & PARSING HELPERS
# =========================================================================
@dataclass(frozen=True)
class StationarySession:
    session_id: str
    heading_deg: float
    t_seconds: np.ndarray       
    gyro: np.ndarray            
    earth_raw: np.ndarray       

def _heading_from_name(path: str) -> float:
    name = Path(path).stem
    nums = re.findall(r"[-+]?\d*\.?\d+", name)
    if not nums:
        return 0.0
    return float(nums[-1]) % 360.0

def _to_seconds(series: pd.Series, time_unit: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        t = series.to_numpy(dtype=np.float64)
        if time_unit in {"s", "ms", "us", "ns"}:
            scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[time_unit]
            t = t * scale
        return t - float(t[0])
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    t_ns = dt.view("int64").to_numpy()
    return (t_ns - int(t_ns[0])).astype(np.float64) * 1e-9

def _resample_matrix(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if t.size < 2: return t.copy(), x.copy()
    order = np.argsort(t)
    t, x = t[order], x[order]
    _, uniq_last_idx = np.unique(t[::-1], return_index=True)
    keep = np.sort((t.size - 1) - uniq_last_idx)
    t, x = t[keep], x[keep]

    dt = 1.0 / fs
    t_u = float(t[0]) + dt * np.arange(int(math.floor((float(t[-1]) - float(t[0])) / dt)) + 1)
    y = np.empty((t_u.size, x.shape[1]), dtype=np.float64)
    finite_t = np.isfinite(t)
    for c in range(x.shape[1]):
        mask = finite_t & np.isfinite(x[:, c])
        if mask.sum() < 2:
            y[:, c] = np.nan
        else:
            y[:, c] = np.interp(t_u, t[mask], x[mask, c])
    return t_u, y


# =========================================================================
# 2. NOISE SYNTHESIS (PSD FDM)
# =========================================================================
def generate_local_synthetic_noise(x_clean: np.ndarray, fs: float, rng: np.random.Generator, window_len: int) -> np.ndarray:
    """
    Extracts local PSD and synthesizes noise via FDM dynamically sized by window_len.
    """
    n_samples = x_clean.shape[0]
    synthetic_noise = np.zeros_like(x_clean)
    
    # Robust Logic: Always split the total window into "n" segments (n_samples // "n").
    # We add a floor of 16 to ensure valid FFT resolution for small windows.
    segment_length = max(n_samples // 4, 16)

    for axis in range(3):
        # Dynamically size the PSD segment length based on the configured window_len
        #segment_length = min(window_len, n_samples) 
        
        f_local, psd_local = welch(x_clean[:, axis], fs=fs, nperseg=segment_length)

        f_grid = np.fft.rfftfreq(n_samples, d=1.0/fs)
        log_psd_interp = np.interp(f_grid, f_local, np.log10(psd_local + 1e-300))
        psd_interp = 10 ** log_psd_interp
        
        mag = np.sqrt(psd_interp * fs * n_samples / 2.0)
        phase = np.exp(1j * 2.0 * np.pi * rng.random(len(f_grid)))
        mag[0] = 0.0 
        
        synthetic_noise[:, axis] = np.fft.irfft(mag * phase, n=n_samples)
        
    return synthetic_noise


# =========================================================================
# 3. DATASET UTILITIES
# =========================================================================
def _compute_stats(sessions: Sequence[StationarySession]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([s.gyro for s in sessions], axis=0)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0, ddof=1)
    return mean.astype(np.float64), std.astype(np.float64)

def _iter_windows(x: np.ndarray, window_len: int, hop_len: int) -> Iterable[tuple[int, np.ndarray]]:
    start = 0
    while start + window_len <= x.shape[0]:
        yield start, x[start:start + window_len]
        start += hop_len

def _split_sessions(sessions: Sequence[StationarySession], seed: int, train_frac: float) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(sessions))
    rng.shuffle(order)
    
    if len(sessions) == 1:
        return {"train": [sessions[0].session_id], "test": []}
        
    n_train = max(1, int(round(train_frac * len(sessions))))
    n_test = len(sessions) - n_train
    
    # Failsafe: Ensure at least one test session if there are multiple sessions available
    if n_test <= 0 and len(sessions) > 1: 
        n_train -= 1
        
    return {
        "train": [sessions[i].session_id for i in order[:n_train]],
        "test": [sessions[i].session_id for i in order[n_train:]],
    }

def _write_index(rows: Sequence[dict], path: Path) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================================================================
# 4. DIAGNOSTICS & EVALUATION
# =========================================================================
def compute_overlapping_adev(y: np.ndarray, fs: float):
    if _HAVE_ALLANTOOLS:
        try:
            taus, adevs, _, _ = allantools.oadev(y, rate=fs, data_type="freq", taus="octave")
            return np.asarray(taus), np.asarray(adevs)
        except Exception:
            pass
    N = len(y)
    max_m = int(np.floor(N / 10))
    m_vals = np.unique(np.logspace(0, np.log10(max_m), 60).astype(int))
    taus, adevs = [], []
    theta = np.cumsum(y) / fs
    for m in m_vals:
        if m == 0: continue
        tau = m / fs
        diff = theta[2*m:] - 2*theta[m:-m] + theta[:-2*m]
        avar = np.sum(diff**2) / (2 * tau**2 * (N - 2*m))
        taus.append(tau)
        adevs.append(np.sqrt(avar))
    return np.array(taus), np.array(adevs)

def run_diagnostic_plot(session_id: str, heading: float, raw_signal: np.ndarray, clean_signal: np.ndarray, synthetic_noise: np.ndarray, noisy_signal: np.ndarray, fs: float, window_len: int, output_dir: Path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Dataset: {session_id} | Heading: {heading}° | Local PSD Synthesis Evaluation", fontsize=16, fontweight='bold')
    axes_names = ['X-Axis', 'Y-Axis', 'Z-Axis']
    time_axis = np.arange(len(raw_signal)) / fs

    for i in range(3):
        # 1. ADEV: Clean vs Synthetic Noise
        t_orig, adev_orig = compute_overlapping_adev(clean_signal[:, i], fs)
        t_syn, adev_syn = compute_overlapping_adev(synthetic_noise[:, i], fs)
        ax_ad = axes[i, 0]
        ax_ad.loglog(t_orig, adev_orig, lw=2.0, label='Clean Signal (Real Noise)', color='tab:blue')
        ax_ad.loglog(t_syn, adev_syn, lw=1.5, ls='--', label='Synthetic Noise', color='tab:orange')
        ax_ad.set_ylabel(f"{axes_names[i]}\nADEV")
        if i == 0: ax_ad.set_title("Allan Deviation")
        if i == 2: ax_ad.set_xlabel("Averaging Time τ (s)")
        ax_ad.grid(True, which='both', ls='--', alpha=0.4)
        if i == 0: ax_ad.legend(fontsize=9)

        # 2. PSD: Clean vs Synthetic Noise
        # Linked directly to the synthetic noise generation logic
        n_samples = len(clean_signal[:, i])
        segment_length = max(n_samples // 4, 16)
        
        f_o, psd_o = welch(clean_signal[:, i], fs=fs, nperseg=segment_length)
        f_s, psd_s = welch(synthetic_noise[:, i], fs=fs, nperseg=segment_length)

        ax_psd = axes[i, 1]
        ax_psd.loglog(f_o[1:], psd_o[1:], lw=2.0, label='Clean Signal (Real Noise)', color='tab:blue')
        ax_psd.loglog(f_s[1:], psd_s[1:], lw=1.5, ls='--', label='Synthetic Noise', color='tab:orange')
        if i == 0: ax_psd.set_title("Power Spectral Density")
        if i == 2: ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.grid(True, which='both', ls='--', alpha=0.4)
        if i == 0: ax_psd.legend(fontsize=9)

        # 3. Time Domain: Raw vs Noisy
        ax_time = axes[i, 2]
        ax_time.plot(time_axis, raw_signal[:, i], lw=1.5, label='Raw Signal', color='tab:blue', alpha=0.7)
        ax_time.plot(time_axis, noisy_signal[:, i], lw=1.0, ls='-', label='Noisy Signal', color='tab:orange', alpha=0.7)
        if i == 0: ax_time.set_title("Time Domain Comparison")
        if i == 2: ax_time.set_xlabel("Time (s)")
        ax_time.grid(True, which='both', ls='--', alpha=0.4)
        if i == 0: ax_time.legend(fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    plt.savefig(output_dir / "diagnostic_psd_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# =========================================================================
# 5. MAIN EXECUTION
# =========================================================================
def main():
    # ---------------------------------------------------------
    # PARAMETER CONFIGURATION 
    # ---------------------------------------------------------
    csv_folder = r"E:\projects\INS\selfSupervised\2025\dataset\datausingTilt\withTilt\fs2000"
    output_dir = r"E:\projects\INS\selfSupervised\2025\Yaser_v1\Stationary_NPY_fs2000"
    
    fs = 100.0
    window_len = 4000          # This parameter explicitly dictates chunk size AND PSD resolution
    hop_len = 1500
    train_frac = 0.8
    
    time_col = "time"
    time_unit = "auto"
    gyro_cols = ["W_x", "W_y", "W_z"]
    earth_cols = ["earth_rate_x", "earth_rate_y", "earth_rate_z"]
    
    seed = 42
    noise_seed_offset = 12345

    # ADD THIS LINE: Set to 20.0, 90.0, etc., or None to just pick the first one
    diagnostic_heading_target = 240  
    # ---------------------------------------------------------

    print("=" * 60)
    print("  Unified Stationary PSD Dataset Pipeline")
    print(f"  Configuration: fs={fs}Hz | window_len={window_len} | hop_len={hop_len}")
    print("=" * 60)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    if not csv_paths:
        raise SystemExit(f"No CSVs matched in: {csv_folder}")

    # 1. Read & Process Raw CSVs (Gyro & Earth Rate directly)
    sessions: list[StationarySession] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        t = _to_seconds(df[time_col], time_unit=time_unit)
        gyro = df[gyro_cols].to_numpy(dtype=np.float64)
        earth_rate = df[earth_cols].to_numpy(dtype=np.float64)

        finite_t = np.isfinite(t)
        t, gyro, earth_rate = t[finite_t], gyro[finite_t], earth_rate[finite_t]

        t_raw = t
        t, gyro = _resample_matrix(t_raw, gyro, fs=fs)
        _, earth_rate = _resample_matrix(t_raw, earth_rate, fs=fs)

        heading = _heading_from_name(path)
        earth_raw_mean = np.nanmean(earth_rate, axis=0) 
        
        sessions.append(StationarySession(Path(path).stem, heading, t, gyro, earth_raw_mean))

    # 2. Split Sessions & Compute Stats
    splits = _split_sessions(sessions, seed, train_frac)
    train_sessions = [s for s in sessions if s.session_id in splits["train"]]
    mean, std = _compute_stats(train_sessions)

    stats = {
        "stationary_earth_dataset": True,
        "fs": fs,
        "window_len": window_len,
        "hop_len": hop_len,
        "train_mean": mean.tolist(),
        "train_std": std.tolist()
    }
    (out_path / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_path / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # 3. Generate Windows & Apply FDM PSD Noise
    written = {"train": 0, "test": 0}
    index_rows: dict[str, list[dict]] = {"train": [], "test": []}
    
    diagnostic_executed = False

    for sess in sessions:
        split = "train" if sess.session_id in splits["train"] else "test"
        earth_norm = (sess.earth_raw - mean) / (std + 1e-12)
        
        # --- Diagnostic Extraction (Linked to window_len) ---
        is_target_heading = True
        if diagnostic_heading_target is not None:
            is_target_heading = abs(sess.heading_deg - diagnostic_heading_target) < 1e-3
            
        # Update: use window_len instead of fixed 4 seconds
        if not diagnostic_executed and is_target_heading and len(sess.gyro) > window_len:
            diag_samples = window_len 
            raw_diag = sess.gyro[:diag_samples]
            clean_diag = raw_diag - sess.earth_raw
            rng_diag = np.random.default_rng(seed)
            
            # This now uses the same window_len as the training loops
            synth_noise_diag = generate_local_synthetic_noise(clean_diag, fs=fs, rng=rng_diag, window_len=window_len)
            noisy_diag = raw_diag + synth_noise_diag
            
            print(f"\n[Diagnostic] Generating localized PSD comparison plot for {sess.session_id} (Heading: {sess.heading_deg}°)...")
            run_diagnostic_plot(
                session_id=sess.session_id, 
                heading=sess.heading_deg, 
                raw_signal=raw_diag, 
                clean_signal=clean_diag, 
                synthetic_noise=synth_noise_diag,
                noisy_signal=noisy_diag,
                fs=fs,
                window_len=window_len,
                output_dir=out_path
            )
            print(f"Diagnostic plot saved to: {out_path / 'diagnostic_psd_evaluation.png'}")
            diagnostic_executed = True
        
        # --- Training Data Generation ---
        for start, win_raw in _iter_windows(sess.gyro, window_len, hop_len):
            if not np.isfinite(win_raw).all(): continue

            key = f"{sess.session_id}|{split}|{start}".encode("utf-8")
            h = int(zlib.crc32(key) & 0xFFFFFFFF)
            rng_win = np.random.default_rng(int((seed + noise_seed_offset + h) % (2**32 - 1)))
            
            win_clean = win_raw - sess.earth_raw
            
            synth_noise = generate_local_synthetic_noise(win_clean, fs=fs, rng=rng_win, window_len=window_len)
            win_noisy_raw = win_raw + synth_noise
            
            clean_norm = ((win_raw - mean) / (std + 1e-12)).T.astype(np.float32) 
            noisy_norm = ((win_noisy_raw - mean) / (std + 1e-12)).T.astype(np.float32) 
            
            idx = written[split]
            fname = f"{sess.session_id}__{split}__{idx:08d}.npy"
            
            for subdir, arr in (("clean", clean_norm), ("noisy", noisy_norm), ("earth", earth_norm.astype(np.float32))):
                path = out_path / split / subdir / fname
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)

            index_rows[split].append({
                "file": fname,
                "session_id": sess.session_id,
                "heading_deg": sess.heading_deg,
                "split": split,
                "start_in_session": int(start),
                "window_len": window_len,
                "hop_len": hop_len,
                "earth_rate_x": float(sess.earth_raw[0]),
                "earth_rate_y": float(sess.earth_raw[1]),
                "earth_rate_z": float(sess.earth_raw[2])
            })
            written[split] += 1

    for split in ("train", "test"):
        _write_index(index_rows[split], out_path / f"index_{split}.csv")

    print("\nDataset Generation Complete:")
    print(json.dumps(written, indent=2))
    print("\n[Done] High-fidelity stationary training pairs successfully generated.")

if __name__ == "__main__":
    main()