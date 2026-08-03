"""Denoise a long 100 Hz CSV session using the DE model (no ORE, no MoE fusion).

This runs the same DE forward path used in training:
[`MOE_mae/DE.py`](../DE.py:1) [`DualMaskMAE.forward()`](../DE.py:227)

Key points:
  - The DE model expects windows of length 256 samples by default.
  - Use overlap-add to reduce boundary artifacts when denoising multi-hour streams.
  - Normalization must match training; this script can read stats.json produced by
    [`MOE_mae/tools/prepare_csv_denoise_dataset.py`](prepare_csv_denoise_dataset.py:1).

Example:
  python MOE_mae/tools/infer_denoise_csv.py \
    --checkpoint runs/de/checkpoint-40.pth \
    --stats_json data/denoise_ds/stats.json \
    --input_csv data/raw/session_01.csv \
    --channel_col gyro_z \
    --time_col timestamp \
    --output_csv outputs/session_01_denoised.csv \
    --device cuda \
    --hop_len 128
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from DE import denoise_dualmask


OMEGA_EARTH_DEGPS = 7.292115e-5 * 180.0 / math.pi


def _to_seconds_from_any_time(series: pd.Series, time_unit: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        t = series.to_numpy(dtype=np.float64)
        if time_unit in {"s", "ms", "us", "ns"}:
            scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[time_unit]
            t = t * scale
        t0 = float(t[0])
        return t - t0

    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.isna().all():
        raise ValueError("Failed to parse time column as numeric or datetime")
    t_ns = dt.view("int64").to_numpy()
    t_ns0 = int(t_ns[0])
    return (t_ns - t_ns0).astype(np.float64) * 1e-9


def _resample_to_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if t.size < 2:
        return t.copy(), x.copy()
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    # drop duplicate timestamps (keep last)
    _, uniq_last_idx = np.unique(t[::-1], return_index=True)
    keep = (t.size - 1) - uniq_last_idx
    keep = np.sort(keep)
    t = t[keep]
    x = x[keep]

    dt_target = 1.0 / fs
    t0 = float(t[0])
    t1 = float(t[-1])
    n = int(math.floor((t1 - t0) / dt_target)) + 1
    t_u = t0 + dt_target * np.arange(n, dtype=np.float64)
    if x.ndim == 1:
        mask = np.isfinite(t) & np.isfinite(x)
        if mask.sum() < 2:
            return t_u, np.full_like(t_u, np.nan, dtype=np.float64)
        x_u = np.interp(t_u, t[mask], x[mask]).astype(np.float64)
        return t_u, x_u

    x_u = np.empty((t_u.size, x.shape[1]), dtype=np.float64)
    finite_t = np.isfinite(t)
    for c in range(x.shape[1]):
        mask = finite_t & np.isfinite(x[:, c])
        if mask.sum() < 2:
            x_u[:, c] = np.nan
        else:
            x_u[:, c] = np.interp(t_u, t[mask], x[mask, c]).astype(np.float64)
    return t_u, x_u


def _heading_from_name(path: str) -> float:
    nums = re.findall(r"[-+]?\d*\.?\d+", Path(path).stem)
    if not nums:
        raise ValueError(f"Could not parse heading angle from filename: {path}")
    return float(nums[-1]) % 360.0


def _parse_cols(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _r_n2b(phi: float, theta: float, psi: float) -> np.ndarray:
    rx = np.array([[1, 0, 0],
                   [0, math.cos(phi), math.sin(phi)],
                   [0, -math.sin(phi), math.cos(phi)]], dtype=np.float64)
    ry = np.array([[math.cos(theta), 0, -math.sin(theta)],
                   [0, 1, 0],
                   [math.sin(theta), 0, math.cos(theta)]], dtype=np.float64)
    rz = np.array([[math.cos(psi), math.sin(psi), 0],
                   [-math.sin(psi), math.cos(psi), 0],
                   [0, 0, 1]], dtype=np.float64)
    return rx @ ry @ rz


def _tilt_rad_from_accel_zup(accel_zup: np.ndarray) -> tuple[float, float]:
    fx, fy, fz = np.asarray(accel_zup, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    phi = math.atan2(fy, fz)
    theta = math.atan2(-fx, math.sqrt(fy * fy + fz * fz))
    return phi, theta


def _stationary_earth_target_raw(
    input_csv: str,
    df: pd.DataFrame,
    stats: dict,
    heading_deg: float | None,
    lat_deg: float | None,
    accel_cols: list[str],
) -> np.ndarray:
    lat = float(lat_deg if lat_deg is not None else stats["lat_deg"])
    heading = float(heading_deg if heading_deg is not None else _heading_from_name(input_csv))
    axis_order = tuple(int(v) for v in stats.get("axis_order", [0, 1, 2]))

    if accel_cols and all(c in df.columns for c in accel_cols):
        accel = df[accel_cols].to_numpy(dtype=np.float64)
        accel_zup = accel[:, list(axis_order)]
        phi, theta = _tilt_rad_from_accel_zup(accel_zup)
    else:
        phi, theta = 0.0, 0.0

    lat_rad = math.radians(lat)
    psi = math.radians(heading)
    omega_nav = np.array([
        OMEGA_EARTH_DEGPS * math.cos(lat_rad),
        0.0,
        OMEGA_EARTH_DEGPS * math.sin(lat_rad),
    ], dtype=np.float64)
    target_zup = _r_n2b(phi, theta, psi) @ omega_nav
    target_original = np.empty(3, dtype=np.float64)
    target_original[list(axis_order)] = target_zup
    return target_original


def load_model(checkpoint_path: str, device: torch.device, in_chans: int = 1, seq_len: int = 256) -> torch.nn.Module:
    model = denoise_dualmask(in_chans=in_chans, seq_len=seq_len).to(device)

    # PyTorch 2.6 changed `torch.load` default `weights_only` from False -> True.
    # Our training checkpoints may store extra metadata (e.g. argparse.Namespace),
    # which can raise an UnpicklingError when `weights_only=True`.
    #
    # We first try the safer `weights_only=True` path with an allowlist for
    # `argparse.Namespace`. If that still fails, we fall back to
    # `weights_only=False` (unsafe for untrusted checkpoints).
    ckpt = None
    try:
        # Newer PyTorch.
        from torch.serialization import safe_globals  # type: ignore

        with safe_globals([argparse.Namespace]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch: no weights_only arg / no safe_globals.
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        warnings.warn(
            "Safe weights-only load failed; retrying with weights_only=False. "
            "Only do this with trusted checkpoints.\n"
            f"Original error: {type(e).__name__}: {e}",
            RuntimeWarning,
        )
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)

    assert ckpt is not None
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def denoise_stream_overlap_add(
    x: np.ndarray,
    model: torch.nn.Module,
    mean: np.ndarray | float,
    std: np.ndarray | float,
    window_len: int,
    hop_len: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Windowed inference with overlap-add + Hann weighting.

    Accepts either a single channel [N] or multiple channels [N, C].
    """
    if window_len % 2 != 0:
        raise ValueError("window_len should be even")
    if hop_len <= 0 or hop_len > window_len:
        raise ValueError("hop_len must be in (0, window_len]")

    x = np.asarray(x, dtype=np.float64)
    input_was_1d = x.ndim == 1
    if input_was_1d:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected x shape [N] or [N,C], got {x.shape}")

    n = int(x.shape[0])
    c = int(x.shape[1])
    if n == 0:
        return x[:, 0].copy() if input_was_1d else x.copy()

    mean_arr = np.asarray(mean, dtype=np.float64).reshape(-1)
    std_arr = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean_arr.size == 1 and c > 1:
        mean_arr = np.repeat(mean_arr, c)
    if std_arr.size == 1 and c > 1:
        std_arr = np.repeat(std_arr, c)
    if mean_arr.size != c or std_arr.size != c:
        raise ValueError(f"Stats length mismatch: got mean={mean_arr.size}, std={std_arr.size}, channels={c}")

    # Normalize.
    x_norm = (x - mean_arr.reshape(1, c)) / (std_arr.reshape(1, c) + 1e-12)

    # Pad so we can process the tail with one final window.
    if n < window_len:
        pad = window_len - n
    else:
        rem = (n - window_len) % hop_len
        pad = 0 if rem == 0 else (hop_len - rem)
    x_pad = np.pad(x_norm, ((0, pad), (0, 0)), mode="reflect")
    n_pad = int(x_pad.shape[0])

    # Hann window for overlap-add.
    w = np.hanning(window_len).astype(np.float64)
    w = np.maximum(w, 1e-6)
    out = np.zeros((n_pad, c), dtype=np.float64)
    weight = np.zeros((n_pad, 1), dtype=np.float64)

    # Build window start indices.
    starts = list(range(0, n_pad - window_len + 1, hop_len))

    # Batch windows.
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        batch = np.stack([x_pad[s : s + window_len] for s in batch_starts], axis=0)  # [B,L,C]
        t = torch.from_numpy(batch).float().permute(0, 2, 1).to(device)  # [B,C,L]
        # DE forward signature is (x_noisy, x_clean). For inference, pass x as both.
        _, y, _, _ = model(t, t)
        y = y.detach().cpu().numpy().astype(np.float64).transpose(0, 2, 1)  # [B,L,C]

        for b, s in enumerate(batch_starts):
            out[s : s + window_len] += y[b] * w[:, None]
            weight[s : s + window_len] += w[:, None]

    y_norm = out / (weight + 1e-12)
    y = y_norm * (std_arr.reshape(1, c) + 1e-12) + mean_arr.reshape(1, c)
    y = y[:n]
    return y[:, 0] if input_was_1d else y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("infer_denoise_csv")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--stats_json", type=str, required=True)
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)

    p.add_argument("--channel_col", type=str, default="", help="single channel to denoise")
    p.add_argument("--channel_cols", type=str, default="", help="comma-separated channels for multi-axis denoising")
    p.add_argument("--time_col", type=str, default="")
    p.add_argument("--time_unit", type=str, default="auto", choices=["auto", "s", "ms", "us", "ns"])
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--allow_resample", action="store_true")

    p.add_argument("--window_len", type=int, default=256)
    p.add_argument("--hop_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--stationary_mean_correction", action="store_true",
                   help="after denoising, shift the session mean to the theoretical stationary Earth-rate vector")
    p.add_argument("--heading_deg", type=float, default=None,
                   help="heading in degrees for stationary mean correction; default parses from input filename")
    p.add_argument("--lat", type=float, default=None,
                   help="latitude in degrees for stationary mean correction; default reads stats_json lat_deg")
    p.add_argument("--accel_cols", type=str, default="A_x,A_y,A_z",
                   help="accelerometer columns used for roll/pitch leveling in stationary mean correction")
    return p.parse_args()


def _channels_from_args(args: argparse.Namespace, stats: dict) -> list[str]:
    if args.channel_cols.strip():
        return [c.strip() for c in args.channel_cols.split(",") if c.strip()]
    if args.channel_col.strip():
        return [args.channel_col.strip()]
    for key in ("channel_cols", "gyro_cols"):
        if key in stats:
            value = stats[key]
            if isinstance(value, list):
                return [str(c) for c in value]
            if isinstance(value, str):
                return [c.strip() for c in value.split(",") if c.strip()]
    raise SystemExit("Provide --channel_col or --channel_cols, or use a stats.json containing channel_cols")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    stats = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))
    mean = np.asarray(stats["train_mean"], dtype=np.float64)
    std = np.asarray(stats["train_std"], dtype=np.float64)
    channel_cols = _channels_from_args(args, stats)

    df = pd.read_csv(args.input_csv)
    missing_cols = [c for c in channel_cols if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing columns {missing_cols} in {args.input_csv}")

    time_col = args.time_col.strip() or None
    if time_col is not None and time_col not in df.columns:
        raise SystemExit(f"Missing time_col {time_col} in {args.input_csv}")

    x = df[channel_cols].to_numpy(dtype=np.float64)
    if len(channel_cols) == 1:
        x = x[:, 0]

    if time_col is None:
        t = np.arange(x.size, dtype=np.float64) / float(args.fs)
    else:
        t = _to_seconds_from_any_time(df[time_col], time_unit=("" if args.time_unit == "auto" else args.time_unit))

    if args.allow_resample and time_col is not None:
        t_u, x_u = _resample_to_uniform(t, x, fs=float(args.fs))
        t, x = t_u, x_u

    model = load_model(
        args.checkpoint,
        device=device,
        in_chans=len(channel_cols),
        seq_len=int(args.window_len),
    )
    y = denoise_stream_overlap_add(
        x=x,
        model=model,
        mean=mean,
        std=std,
        window_len=int(args.window_len),
        hop_len=int(args.hop_len),
        device=device,
        batch_size=int(args.batch_size),
    )

    if args.stationary_mean_correction:
        if len(channel_cols) != 3:
            raise SystemExit("--stationary_mean_correction requires exactly three gyro channels")
        target = _stationary_earth_target_raw(
            input_csv=args.input_csv,
            df=df,
            stats=stats,
            heading_deg=args.heading_deg,
            lat_deg=args.lat,
            accel_cols=_parse_cols(args.accel_cols),
        )
        if np.asarray(y).ndim != 2 or y.shape[1] != 3:
            raise SystemExit("stationary mean correction expected denoised output shape [N,3]")
        correction = target.reshape(1, 3) - y.mean(axis=0, keepdims=True)
        y = y + correction
        print("Applied stationary mean correction (deg/s):", correction.reshape(-1).tolist())

    out_data = {(time_col or "t_seconds"): t}
    if len(channel_cols) == 1:
        out_data[channel_cols[0]] = x
        out_data[f"{channel_cols[0]}_denoised"] = y
    else:
        for i, col in enumerate(channel_cols):
            out_data[col] = x[:, i]
            out_data[f"{col}_denoised"] = y[:, i]
    out_df = pd.DataFrame(out_data)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()

