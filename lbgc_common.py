"""
Learning-Based MEMS Gyrocompassing (LBGC) -- shared utilities.

This module is the fixed, local (non-Colab) core used by both `lbgc_train.py`
and `lbgc_inference.py`. It adapts the original notebooks to the local dataset:

Fixes applied vs. the original notebooks
----------------------------------------
* Sampling rate is 2000 Hz (was 600 Hz).
* Latitude is read from `Dataset/stat.txt` (e.g. "lat:35.727") instead of being
  hard-coded.
* CSV columns are `time, W_x, W_y, W_z, A_x, A_y, A_z` (gyro in deg/s,
  accelerometer in m/s**2). The original code expected `w_x..f_z`.
* Ground-truth heading is parsed from the file name `Psi_<angle>.csv`
  (the original parsed `x_<angle>.csv`).
* All Google Colab / Google Drive code is removed; paths point at the local
  `Dataset/` folder next to this file.
* Downsampling is done with exact-length, anti-aliasing block averaging, which
  is robust for the large decimation factor used here (the original relied on
  `scipy.signal.decimate(q=500)`, which is numerically unstable and produced
  length-mismatch bugs).
"""

import os
import re
import glob
import random

import numpy as np
import torch
import torch.nn as nn


# ===================================================== #
#                     Paths                             #
# ===================================================== #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
STAT_FILE = os.path.join(DATASET_DIR, "stat.txt")

MODEL_PATH = os.path.join(BASE_DIR, "Model_Best.pt")
META_PATH = os.path.join(BASE_DIR, "Model_Best_meta.json")

# --------------------------------------------------------------------------- #
# Sensor axis convention.
#
# This dataset uses a Y-up frame: the vertical axis is Y, i.e. gyro W_y carries
# Earth's vertical rate (w_ie*sin(lat)) and accel A_y carries gravity (~9.8).
# The gyrocompassing formula (GC_Mean) and the yaw augmentation (rotation about
# Z) both assume a Z-up frame. We therefore reorder columns (X, Y, Z) -> (X, Z, Y)
# at load time so Z becomes the vertical axis. With this swap the analytical GC
# baseline drops from ~77 deg RMSE to ~0.85 deg RMSE (verified), and the yaw
# augmentation rotates the correct (horizontal) plane.
# Set AXIS_ORDER = (0, 1, 2) to disable the remap.
# --------------------------------------------------------------------------- #
AXIS_ORDER = (0, 2, 1)


# ===================================================== #
#                   Device / precision                  #
# ===================================================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Navigation-grade IMU: the signal (Earth rate ~0.003 deg/s) and the errors are
# tiny, so the whole DL pipeline runs in double precision (float64) to avoid
# discarding significant digits. Set to torch.float32 for speed if desired.
DTYPE = torch.float64

# Use accelerometer leveling (roll/pitch) in the analytical gyrocompassing
# baseline. This platform is tilted ~1.45 deg; leveling removes the resulting
# heading error and takes the baseline from ~0.85 deg to ~0.09 deg RMSE.
USE_LEVELING = True


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ===================================================== #
#                    Configuration                      #
# ===================================================== #
class CFG:
    # General
    Deg_2_Rad = np.pi / 180.0            # [rad/deg]
    Rad_2_Deg = 180.0 / np.pi            # [deg/rad]
    hour = 3600                          # [s]
    DegHR_2_RadSEC = np.pi / 180 / hour  # deg/hour -> rad/sec
    mRad = 1000

    freq = 2000                          # [Hz] Sensor sampling rate (fixed for this dataset)
    dt = 1.0 / freq                      # [sec]
    fig_size = (8, 5)


# Geodetic params
Re = 6378137                            # [m]
ecc = 0.0818191908426215               # Earth eccentricity, e2 = 2*f - f^2
E_2 = ecc ** 2                          # squared eccentricity
w_ie = 7292115e-11                      # Earth's rotation rate [rad/s]  (7.292115e-5)
W_ie = [0, 0, w_ie]

rad_2_deg = lambda x: x * (180 / np.pi)
deg_2_rad = lambda x: x * (np.pi / 180)
mRad_2_deg = lambda x: x * (180 / np.pi) / 1000
deg_2_mrad = lambda x: deg_2_rad(x) * 1000
Sec_2_hr = lambda x: x * CFG.hour
Sec_2_day = lambda x: Sec_2_hr(x) * 24

f_norm = lambda x: np.sqrt(x @ x)
f_normalize = lambda x: (x - x.mean()) / (x.max() - x.min())


def load_latitude(stat_file=STAT_FILE):
    """Read latitude [deg] from stat.txt (format like 'lat:35.727')."""
    with open(stat_file, "r") as fh:
        text = fh.read()
    m = re.search(r"lat\s*[:=]\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not parse latitude from {stat_file!r}: {text!r}")
    return float(m.group(1))


# Expected Earth-rate magnitude [deg/hr] for a quick sanity check (~15.041 deg/hr)
EARTH_RATE_DEG_HR = rad_2_deg(w_ie) * CFG.hour


# ===================================================== #
#                 Coarse alignment                      #
# ===================================================== #
def R_b2n(phi, theta, psi):
    """Euler angles from Body to Navigation DCM."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0, 1, 0],
                   [np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    return Rz.T @ Ry.T @ Rx.T


def R_n2b(phi, theta, psi):
    """Euler angles from Navigation to Body DCM."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0, 1, 0],
                   [np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    return Rx @ Ry @ Rz


def R_e2n(lat, long):
    """Global frame (ECEF) to Navigation frame (NED/LLN)."""
    return np.array(
        [[-np.sin(lat) * np.cos(long), -np.sin(lat) * np.sin(long), np.cos(lat)],
         [-np.sin(long), np.cos(long), 0],
         [-np.cos(lat) * np.cos(long), -np.cos(lat) * np.sin(long), -np.sin(lat)]])


def tilt_rad(f_ib):
    """Roll (phi) and pitch (theta) in RADIANS from specific force.

    This dataset (after the Z-up axis remap) reports A_z ~ +9.8 m/s**2, i.e.
    the accelerometer reads +g on the up axis. The tilt is derived from the
    mean specific force accordingly (up-positive convention).
    """
    fx, fy, fz = np.asarray(f_ib).reshape(-1, 3).mean(0)
    phi = np.arctan2(fy, fz)                          # roll  about x
    theta = np.arctan2(-fx, np.sqrt(fy ** 2 + fz ** 2))  # pitch about y
    return phi, theta


def levelling(f_ib):
    """Roll (phi) and pitch (theta) in DEGREES (human-readable diagnostic)."""
    phi, theta = tilt_rad(f_ib)
    return rad_2_deg(phi), rad_2_deg(theta)


# ===================================================== #
#                   Gyrocompassing                      #
# ===================================================== #
def GC_Mean(Gyro, f_ib=None):
    """Analytical (Groves) gyrocompassing baseline.

    Input : noisy gyroscope measurement R^{n x 3} in [deg/s]; optional specific
            force R^{n x 3} in [m/s**2] for accelerometer leveling.
    Output: estimated heading angle [deg] in [0, 360)

    When `f_ib` is given the roll/pitch tilt is removed (leveled solution),
    which is essential for a tilted navigation-grade platform. `phi`/`theta`
    are handled internally in RADIANS (the original notebook passed degrees into
    np.cos/np.sin -- a latent bug that only stayed hidden because it always ran
    with tilt disabled).
    """
    gyr = Gyro.mean(0) * CFG.Deg_2_Rad          # convert to rad/s and average
    if f_ib is None:
        phi, theta = 0.0, 0.0
    else:
        phi, theta = tilt_rad(f_ib)             # radians

    sin_psi = -gyr[1] * np.cos(phi) + gyr[2] * np.sin(phi)
    cos_psi = (gyr[0] * np.cos(theta)
               + gyr[1] * np.sin(phi) * np.sin(theta)
               + gyr[2] * np.cos(phi) * np.sin(theta))
    psi_GC = rad_2_deg(np.arctan2(sin_psi, cos_psi)) % 360
    return psi_GC


def GC_error(GT_i, y_GC):
    """Heading error wrapped to (-180, 180] degrees."""
    err = GT_i - y_GC
    if err > 180:
        err -= 360
    elif err < -180:
        err += 360
    return err


def batch_error(x_input, y_GT, acc=None):
    """RMSE [deg] of the analytical baseline over a batch of samples.

    If `acc` (specific force) is given and USE_LEVELING is True, the leveled
    solution is used.
    """
    err = np.zeros(y_GT.shape[0])
    for i, gt in enumerate(y_GT):
        f_ib = acc[i] if (acc is not None and USE_LEVELING) else None
        y_GC = GC_Mean(x_input[i], f_ib) % 360
        err[i] = GC_error(gt, y_GC)
    return np.sqrt(np.mean(err ** 2))


# ===================================================== #
#                     Data loading                      #
# ===================================================== #
def label_from_path(path):
    """Parse ground-truth heading [deg] from a filename like 'Psi_045.csv'."""
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"[-+]?\d*\.?\d+", name)
    if not nums:
        raise ValueError(f"No angle found in filename: {path!r}")
    return np.float32(nums[-1])


def load_dataset(data_dir, t_len):
    """Load every CSV in `data_dir`, sorted ascending by heading angle.

    Returns
    -------
    x_gyr : (N, t_len, 3) float64   gyro   [deg/s]   (W_x, W_y, W_z)
    x_acc : (N, t_len, 3) float64   accel  [m/s**2]  (A_x, A_y, A_z)
    y     : (N,)          float32   heading [deg]
    paths : list[str]
    """
    import pandas as pd

    paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir!r}")

    labels = np.array([label_from_path(p) for p in paths])
    order = np.argsort(labels)
    paths = [paths[i] for i in order]
    y = labels[order]

    n = len(paths)
    x_gyr = np.zeros((n, t_len, 3))
    x_acc = np.zeros((n, t_len, 3))

    gyro_cols = ["W_x", "W_y", "W_z"]
    acc_cols = ["A_x", "A_y", "A_z"]

    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df = df.loc[:t_len - 1, ~df.columns.str.contains("^Unnamed")]
        df = df.dropna(how="all").reset_index(drop=True)
        if len(df) < t_len:
            raise ValueError(
                f"{os.path.basename(path)} has {len(df)} rows < t_len={t_len}. "
                f"Lower CFG t_len / TIME_LIMIT.")
        x_gyr[i] = df.loc[:t_len - 1, gyro_cols].values.astype(float)
        x_acc[i] = df.loc[:t_len - 1, acc_cols].values.astype(float)

    # Reorder axes to a Z-up frame (see AXIS_ORDER note at top of file).
    x_gyr = x_gyr[:, :, list(AXIS_ORDER)]
    x_acc = x_acc[:, :, list(AXIS_ORDER)]

    return x_gyr, x_acc, y, paths


def print_dataset_summary(x_gyr, y, x_acc=None, tag=""):
    """Print per-file GC baseline diagnostics (mirrors the notebook printout).

    Uses accelerometer leveling when x_acc is supplied and USE_LEVELING is True.
    """
    leveled = x_acc is not None and USE_LEVELING
    print(f"\n--- {tag} ({len(y)} files) | leveled baseline: {leveled} ---")
    for i in range(len(y)):
        g = x_gyr[i].mean(0)
        f_ib = x_acc[i] if leveled else None
        y_gc = GC_Mean(x_gyr[i], f_ib) % 360
        roll, pitch = (levelling(x_acc[i]) if x_acc is not None else (0.0, 0.0))
        print("{:2d}) GT: {:6.1f}  GC: {:9.4f}  err: {:+8.4f}  "
              "(wx,wy,wz)=({:+.5f},{:+.5f},{:+.5f})  |w_ib|: {:7.4f} deg/hr  "
              "roll/pitch: {:+.3f}/{:+.3f}".format(
                  i + 1, y[i], y_gc, GC_error(y[i], y_gc),
                  g[0], g[1], g[2], Sec_2_hr(f_norm(x_gyr[i].mean(0))),
                  roll, pitch))


# ===================================================== #
#            Augmentation & downsampling                #
# ===================================================== #
def downsample(arr, ds_len):
    """Exact-length anti-aliasing downsample by block averaging.

    Parameters
    ----------
    arr    : (N, T, C) array
    ds_len : target sequence length

    Returns
    -------
    (N, ds_len, C) array
    """
    n, t, c = arr.shape
    q = t // ds_len
    if q < 1:
        raise ValueError(f"ds_len={ds_len} larger than sequence length T={t}")
    t_trim = q * ds_len
    return arr[:, :t_trim, :].reshape(n, ds_len, q, c).mean(axis=2)


def Aug_and_DS(arr, y_label, aug_factor, psi_range, ds_len,
               noise_scale=0.1, bias_scale=1e-4, seed=None):
    """Augment (yaw rotation + noise + bias) then downsample.

    For each recording we synthesise `aug_factor` new samples at headings
    (label + psi_range[j]) by rotating the gyro vector about the vertical axis,
    adding white noise (scaled to 10% of each channel's std) and a small bias.
    """
    rng = np.random.default_rng(seed)
    d_1, d_2, d_3 = arr.shape
    x_aug = np.zeros((d_1 * aug_factor, ds_len, d_3))
    y_aug = np.zeros(d_1 * aug_factor)

    for i in range(d_1):
        wn = arr[i].std(0) * rng.standard_normal((d_2, d_3)) * noise_scale
        bias = rng.standard_normal(3) * bias_scale
        for j in range(aug_factor):
            idx = i * aug_factor + j
            y_aug[idx] = (y_label[i] + psi_range[j]) % 360
            x_rot = arr[i] @ R_n2b(0, 0, deg_2_rad(psi_range[j])).T + wn + bias
            x_aug[idx] = downsample(x_rot[None], ds_len)[0]

    return x_aug, y_aug


# ===================================================== #
#                   Loss function                       #
# ===================================================== #
class cyclicMSE(nn.Module):
    """Angle-aware MSE (in deg**2): treats 359 deg and 1 deg as 2 deg apart."""

    def forward(self, y_a, y_b):
        y_a, y_b = deg_2_rad(y_a), deg_2_rad(y_b)
        error = torch.atan2(torch.sin(y_a - y_b), torch.cos(y_a - y_b))
        return torch.mean(rad_2_deg(error) ** 2)


# ===================================================== #
#                   Dataloaders                         #
# ===================================================== #
def x_preprocess(arr, batch_size=None, is_sample=True):
    """Flatten (N, T, C) -> (N, C*T) and split into equal batches.

    Returns a tensor of shape (num_batches, batch_size, C*T) for samples, or
    (num_batches, batch_size) for labels. `batch_size` must divide N; for
    validation/test call with batch_size=None to keep everything in one batch.
    """
    if batch_size is None:
        batch_size = arr.shape[0]

    if is_sample:
        # || AXIS 1: interleave per channel (best per the paper)
        flat = torch.flatten(
            torch.tensor(arr, dtype=DTYPE).permute(0, 2, 1), start_dim=1)
    else:
        flat = torch.tensor(arr, dtype=DTYPE)

    pieces = torch.split(flat, batch_size)
    return torch.stack(list(pieces), dim=0).to(device)


# ===================================================== #
#                   Model zoo                           #
# ===================================================== #
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        b = x.size(0)
        shape = (self.num_layers * self.num_directions, b, self.hidden_size)
        h0 = torch.zeros(shape, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(shape, device=x.device, dtype=x.dtype)
        out, _ = self.lstm(x, (h0, c0))
        pred = self.fc(out).squeeze(0).squeeze(-1)
        return pred % 360


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        b = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, b,
                         self.hidden_size, device=x.device, dtype=x.dtype)
        out, _ = self.gru(x, h0)
        pred = self.fc(out).squeeze(0).squeeze(-1)
        return pred % 360


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        b = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, b,
                         self.hidden_size, device=x.device, dtype=x.dtype)
        out, _ = self.rnn(x, h0)
        pred = self.fc(out).squeeze(0).squeeze(-1)
        return pred % 360


MODEL_NAMES = {1: "LSTM", 2: "GRU", 3: "RNN"}


def build_model(choice, input_size, hidden_size, output_size, num_layers,
                bidirectional=True):
    """Instantiate a model by choice (1=LSTM, 2=GRU, 3=RNN)."""
    args = (input_size, hidden_size, output_size, num_layers)
    if choice == 1:
        model = LSTM(*args, bidirectional=bidirectional)
    elif choice == 2:
        model = GRU(*args, bidirectional=bidirectional)
    elif choice == 3:
        model = RNN(*args, bidirectional=bidirectional)
    else:
        raise ValueError("model_choice must be 1 (LSTM), 2 (GRU) or 3 (RNN)")
    return model.to(device=device, dtype=DTYPE)
