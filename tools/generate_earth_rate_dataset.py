"""
Generate Earth-Rate Physical Baseline Dataset

This script reads raw IMU session CSVs, parses their heading and latitude,
calculates the tilt (roll/pitch) from the accelerometer readings, projects
the theoretical Earth rotation rate onto the 3D body axes of the gyroscope,
and saves the results into a new CSV directory.

Output CSV Columns:
    time, W_x, W_y, W_z, A_x, A_y, A_z, earth_rate_x, earth_rate_y, earth_rate_z
"""

from __future__ import annotations

import glob
import math
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Earth's rotation rate in degrees per second (~15.041 deg/hr)
OMEGA_EARTH_DEGPS = 7.292115e-5 * 180.0 / math.pi


def _heading_from_filename(path: str) -> float:
    """Extract heading angle from filenames like 'Psi_045.csv'."""
    name = Path(path).stem
    nums = re.findall(r"[-+]?\d*\.?\d+", name)
    if not nums:
        raise ValueError(f"Could not parse heading angle from filename: {path}")
    return float(nums[-1]) % 360.0


def _load_latitude(stat_file: str) -> float:
    """Parse latitude from a stat.txt file (e.g., 'lat:35.727')."""
    if not os.path.exists(stat_file):
        raise FileNotFoundError(f"Latitude stat file not found: {stat_file}")
    text = Path(stat_file).read_text(encoding="utf-8")
    m = re.search(r"lat\s*[:=]\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not parse latitude from {stat_file}")
    return float(m.group(1))


def _to_seconds(series: pd.Series, time_unit: str) -> np.ndarray:
    """Standardize time array to elapsed seconds from start."""
    if pd.api.types.is_numeric_dtype(series):
        t = series.to_numpy(dtype=np.float64)
        if time_unit in {"s", "ms", "us", "ns"}:
            scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[time_unit]
            t = t * scale
        return t - float(t[0])

    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.isna().all():
        raise ValueError("Failed to parse time column as numeric or datetime")
    t_ns = dt.view("int64").to_numpy()
    return (t_ns - int(t_ns[0])).astype(np.float64) * 1e-9


def _resample_matrix(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample matrix to a uniform grid via linear interpolation."""
    if t.size < 2:
        return t.copy(), x.copy()

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # Drop duplicate timestamps
    _, uniq_last_idx = np.unique(t[::-1], return_index=True)
    keep = np.sort((t.size - 1) - uniq_last_idx)
    t = t[keep]
    x = x[keep]

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


def _tilt_from_accel(accel_zup: np.ndarray) -> tuple[float, float]:
    """Compute roll (phi) and pitch (theta) from gravity vector (Z-up frame)."""
    fx, fy, fz = np.asarray(accel_zup, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    phi = math.atan2(fy, fz)
    theta = math.atan2(-fx, math.sqrt(fy * fy + fz * fz))
    return phi, theta


def _r_n2b(phi: float, theta: float, psi: float) -> np.ndarray:
    """Rotation matrix from Navigation (NED) to Body frame (Z-up)."""
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


def calculate_earth_rate(
    lat_deg: float,
    heading_deg: float,
    accel: np.ndarray,
    axis_order: tuple[int, int, int],
) -> np.ndarray:
    """Project local Earth rate into the sensor's physical frame."""
    lat = math.radians(lat_deg)
    psi = math.radians(heading_deg)

    # Reorder accelerometer measurements to match the internal Z-up physical model
    accel_zup = accel[:, list(axis_order)]
    phi, theta = _tilt_from_accel(accel_zup)

    # Earth rate in local navigation frame (North, East, Down/Up)
    omega_nav = np.array([
        OMEGA_EARTH_DEGPS * math.cos(lat),
        0.0,
        OMEGA_EARTH_DEGPS * math.sin(lat),
    ], dtype=np.float64)

    # Rotate local rate into body frame (Z-up coordinate configuration)
    target_zup = _r_n2b(phi, theta, psi) @ omega_nav

    # Project the aligned vector back to the original physical axis order
    target_original = np.empty(3, dtype=np.float64)
    target_original[list(axis_order)] = target_zup
    return target_original


def main():
    # =========================================================================
    # 1. DIRECTORIES AND PATHS
    # =========================================================================
    csv_folder = r"E:\Yaser\New Day\Runs\Dataset"
    output_folder = r"E:\Yaser\New Day\Runs\Dataset\earth_rate_baselines"
    stat_file = r"E:\Yaser\New Day\Runs\Dataset\stat.txt"

    # =========================================================================
    # 2. SENSOR & PHYSICAL PARAMETERS
    # =========================================================================
    fs = 100                 # Target sampling rate in Hz
    lat = None                  # Set to float (e.g., 35.727), or leave None to read from stat_file
    axis_order = (0, 2, 1)      # Permutation mapping axes to local Z-up frame

    # =========================================================================
    # 3. CSV COLUMN CONFIGURATIONS
    # =========================================================================
    time_col = "time"           
    time_unit = "auto"          # Options: "auto", "s", "ms", "us", "ns"
    gyro_cols = ["W_x", "W_y", "W_z"]
    accel_cols = ["A_x", "A_y", "A_z"]

    # =========================================================================
    # EXECUTION LOGIC
    # =========================================================================
    
    # Resolve Latitude
    resolved_lat = lat if lat is not None else _load_latitude(stat_file)
    print(f"Loaded configuration: Latitude={resolved_lat}° | Target Sampling Rate={fs}Hz")

    # Create output directory if it doesn't exist
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # Gather all CSV files in the input folder
    search_pattern = os.path.join(csv_folder, "*.csv")
    csv_paths = sorted(glob.glob(search_pattern))
    
    if not csv_paths:
         raise SystemExit(f"No CSV files found in: {csv_folder}")

    for path in csv_paths:
        print(f"Processing: {Path(path).name}")
        df = pd.read_csv(path)

        # Parse time and format arrays
        t = _to_seconds(df[time_col], time_unit=time_unit)
        gyro = df[gyro_cols].to_numpy(dtype=np.float64)
        accel = df[accel_cols].to_numpy(dtype=np.float64)

        # Uniform Resampling
        t_u, gyro_u = _resample_matrix(t, gyro, fs=fs)
        _, accel_u = _resample_matrix(t, accel, fs=fs)

        # Heading Parsing
        heading = _heading_from_filename(path)

        # Calculate exact Earth rate vector matching the raw axes
        earth_rate = calculate_earth_rate(resolved_lat, heading, accel_u, axis_order)

        # Create output DataFrame matching the clean gyro columns, accel columns, and the target physical projections
        out_df = pd.DataFrame({
            time_col: t_u,
            gyro_cols[0]: gyro_u[:, 0],
            gyro_cols[1]: gyro_u[:, 1],
            gyro_cols[2]: gyro_u[:, 2],
            accel_cols[0]: accel_u[:, 0],
            accel_cols[1]: accel_u[:, 1],
            accel_cols[2]: accel_u[:, 2],
            'earth_rate_x': earth_rate[0],
            'earth_rate_y': earth_rate[1],
            'earth_rate_z': earth_rate[2]
        })

        # Drop any records where interpolation resulted in invalid data points
        out_df = out_df.dropna().reset_index(drop=True)

        # Save to target folder
        out_file = out_path / Path(path).name
        out_df.to_csv(out_file, index=False)
        print(f"Saved physical baseline data to: {out_file}\n")

    print(f"Successfully generated baseline datasets in {output_folder}!")


if __name__ == "__main__":
    main()