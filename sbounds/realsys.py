"""Real-data systems: NASA C-MAPSS turbofan fleet as a stochastic physical system.

The C-MAPSS FD001 training split records 21 sensor channels per engine cycle for
100 engines run to failure. We use a per-life-era linear-Gaussian (Ornstein-
Uhlenbeck) identification of the sensor dynamics:

    dx = A x + b dt + Sigma dW,        Sigma = diag(sigma_vec),

fit by least squares on first differences of the selected channels, separately
for a healthy era (early life) and an aged era (late life). This is a *model of
real data*: A, b and the per-channel noise magnitudes are estimated from NASA
trajectories, and the aged-era fit is the plant drift that the shadow gate must
detect -- real distribution shift, not a synthetic damping change.

The identification is local (linear in sensor coordinates). Engine degradation
is a slow trend, which the OU fit absorbs into era-dependent parameters; that is
exactly the structure the hot-swap experiment is designed to test. The Hurwitz
clamp in `_hurwitz_project` keeps the identified model inside the class the
certificate machinery requires and is recorded, not hidden.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# 8 operational channels of FD001 (1-indexed as in the C-MAPSS schema):
# T24, T30, T50, P30, Ps30, phi, NRf, BPR -- the standard informative subset.
SENSOR_COLS = [2, 3, 4, 8, 9, 15, 13, 12]  # 1-indexed within the 3+21 block
SENSOR_NAMES = ["T24", "T30", "T50", "P30", "Ps30", "phi", "NRf", "BPR"]


def _name_of(sensor_col: int) -> str:
    """C-MAPSS schema name of a 1-indexed sensor column, or s<col> if unknown."""
    _schema = {2: "T24", 3: "T30", 4: "T50", 8: "P30", 9: "Ps30",
               15: "phi", 13: "NRf", 12: "BPR", 11: "NRc", 17: "P15",
               7: "T48", 5: "P2", 6: "T2", 16: "epr", 10: "mf",
               18: "egThd", 19: "egTdm", 20: "P40", 21: "P50", 22: "ps30",
               23: "nf", 24: "Nc", 25: "SmP30", 26: "SmP40"}
    return _schema.get(sensor_col, f"s{sensor_col}")


def _hurwitz_project(A: torch.Tensor, margin: float = 0.02, rho_max: float = 0.8) -> tuple[torch.Tensor, float]:
    """Project eigenvalues into the stable disk {Re <= -margin, |lambda| <= rho_max}.

    Least-squares identification of trending sensor data produces stiff, mildly
    unstable eigenvalues; the certificate machinery needs a Hurwitz drift. The
    projection magnitude is returned and recorded by the caller -- it is an
    honest, reported model-class constraint, not a hidden correction.
    """
    w, Vv = torch.linalg.eig(A)
    re = w.real.clamp_max(-margin)
    mag = (re ** 2 + w.imag ** 2).sqrt()
    scale = (rho_max / mag).clamp_max(1.0)
    w2 = torch.complex(re * scale, w.imag * scale)
    A2 = (Vv @ torch.diag(w2) @ torch.linalg.inv(Vv)).real
    return A2, float((A2 - A).abs().max())


class SensorSystem:
    """Ornstein-Uhlenbeck sensor dynamics identified from real fleet data.

    Exposes the same interface as ChainArm: dim, drift, diffusion, equilibrium,
    lqr_like_scale, noise == "diagonal". dtype is float64 throughout.
    """

    noise = "diagonal"

    def __init__(self, A: torch.Tensor, b: torch.Tensor, sigma_vec: torch.Tensor,
                 dt_note: str = ""):
        self.A = A.to(torch.float64)
        self.b = b.to(torch.float64)
        self.sigma_vec = sigma_vec.to(torch.float64)
        self.dt_note = dt_note
        self.clamp_max = 0.0

    # --- construction -------------------------------------------------------
    @classmethod
    def fit(cls, X: torch.Tensor, unit_ids: torch.Tensor | None = None) -> "SensorSystem":
        """Least-squares OU fit on first differences: dx = A x + b (+ noise).

        X is (N, D) consecutive sensor states (cycle-to-cycle, dt = 1 cycle).
        Differences are taken within each unit when unit_ids is provided.
        """
        X = X.to(torch.float64)
        if unit_ids is None:
            X0, X1 = X[:-1], X[1:]
        else:
            keep = unit_ids[1:] == unit_ids[:-1]
            X0, X1 = X[:-1][keep], X[1:][keep]
        Y = X1 - X0                                    # (M, D)
        Z = torch.cat([X0, torch.ones(X0.shape[0], 1, dtype=X0.dtype)], dim=1)
        theta = torch.linalg.lstsq(Z, Y).solution      # (D+1, D)
        A = theta[:-1, :].T                            # drift: dx = A x + b
        b = theta[-1, :]
        resid = Y - Z @ theta
        # per-channel residual std: the estimated diffusion magnitude per cycle
        sigma_vec = resid.std(dim=0, correction=1).clamp_min(1e-6)
        sys = cls(A, b, sigma_vec)
        # the certificate machinery needs a Hurwitz drift; clamp and record
        A2, cmax = _hurwitz_project(A)
        sys.A = A2
        sys.clamp_max = cmax
        return sys

    # --- the System interface ------------------------------------------------
    @property
    def dim(self) -> int:
        return self.A.shape[0]

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + self.b

    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        # (..., D, D) diagonal, constant: Sigma = diag(sigma_vec)
        return torch.diag_embed(self.sigma_vec).expand(*x.shape, x.shape[-1])

    def equilibrium(self) -> torch.Tensor:
        """The OU mean -A^{-1} b: the true attractor of the identified model."""
        return torch.linalg.solve(-self.A, self.b)

    def lqr_like_scale(self) -> torch.Tensor:
        """Per-coordinate stationary std: solve the OU Lyapunov equation for
        Sigma_stat (A S + S A^T + SS^T = 0 with S = diag(sigma_vec)) and take
        sqrt(diag). Used only for random initialisation and region sizing.
        For D > 64 the exact Kronecker solve (a D^2 linear system) is replaced
        by the diagonal approximation sqrt(sigma_i^2 / (-2 Re(A_ii))) -- exact
        for diagonal A, right scaling for weakly coupled ones. Sizing only:
        this never enters a sound bound."""
        D = self.dim
        diag_approx = (self.sigma_vec
                       / torch.sqrt((-2.0 * torch.diagonal(self.A)).clamp_min(1e-6)))\
            .clamp_min(1e-4)
        if D > 64:
            return diag_approx
        try:
            S = torch.diag_embed(self.sigma_vec)
            # vec convention: vec(A X B) = (B^T kron A) vec(X); here A X + X A^T
            K = torch.kron(self.A, torch.eye(D, dtype=self.A.dtype)) + \
                torch.kron(torch.eye(D, dtype=self.A.dtype), self.A)
            sol = torch.linalg.solve(K, (-S.reshape(-1, 1)))
            Sigma_stat = sol.reshape(D, D)
            Sigma_stat = 0.5 * (Sigma_stat + Sigma_stat.T)
            # numerical safety: PD-ify via eigenvalue floor; strongly correlated
            # plants (SARCOS torques) can make K ill-conditioned enough that
            # even eigh refuses -- sizing-only, so fall back to the diagonal
            # approximation rather than crash the pipeline.
            w, Vv = torch.linalg.eigh(Sigma_stat)
            Sigma_stat = Vv @ torch.diag(w.clamp_min(1e-8)) @ Vv.T
            return torch.sqrt(torch.diagonal(Sigma_stat)).clamp_min(1e-4)
        except Exception:
            return diag_approx

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"SensorSystem(D={self.dim}, sigma=[{', '.join(f'{s:.3f}' for s in self.sigma_vec[:4])}, ...], "
                f"clamp={self.clamp_max:.3f})")


def load_cmapss_fd001(data_dir: str | None = None,
                      sensors: list[int] | None = None,
                      fit_units: int = 80) -> dict:
    """Load FD001 and return z-scored per-unit sensor arrays + era splits.

    Channels are standardized with mean/std from the fit units only (no test
    leakage). The identified systems operate in these standardized coordinates.
    Returns dict with:
      X (N,D) float64, unit (N,), era_lo/era_hi (N,) 0/1 flags
      (era_lo = first half of each unit's life, era_hi = second half),
      held-out units as X_hold/unit_hold, and the mu/sd used.
    """
    sensors = sensors or SENSOR_COLS
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(here), "data", "cmapss", "CMaps")
    path = os.path.join(data_dir or default, "train_FD001.txt")
    rows = []
    units = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            units.append(int(parts[0]))
            # cols: unit, cycle, 3 settings, 21 sensors -> 1-indexed sensor col
            rows.append([float(parts[c]) for c in sensors])
    X = torch.tensor(rows, dtype=torch.float64)
    unit = torch.tensor(units, dtype=torch.long)
    fit_mask = unit <= fit_units
    Xfit = X[fit_mask]
    # Drop near-constant channels: in FD001 (single operating condition) some
    # sensors never move. They carry no dynamics, make the least-squares design
    # rank-deficient (non-deterministic pseudo-inverse across LAPACK builds),
    # and would only add dead coordinates to the state.
    raw_sd = Xfit.std(dim=0)
    keep = raw_sd > 1e-3
    dropped = [_name_of(s) for s, k in zip(sensors, keep.tolist()) if not k]
    sensors = [s for s, k in zip(sensors, keep.tolist()) if k]
    X = X[:, keep]
    Xfit = Xfit[:, keep]
    mu = Xfit.mean(dim=0)
    sd = Xfit.std(dim=0).clamp_min(1e-9)
    Xz = (X - mu) / sd
    out = {"X": Xz[fit_mask], "unit": unit[fit_mask],
           "X_hold": Xz[~fit_mask], "unit_hold": unit[~fit_mask],
           "mu": mu, "sd": sd, "dropped_channels": dropped,
           "sensors": sensors, "names": [_name_of(s) for s in sensors]}
    # era split per unit: first half of life vs second half (real aging drift)
    era_lo = torch.zeros_like(out["unit"], dtype=torch.float64)
    era_hi = torch.zeros_like(out["unit"], dtype=torch.float64)
    for u in out["unit"].unique():
        m = out["unit"] == u
        idx = torch.nonzero(m).squeeze(-1)
        half = idx.numel() // 2
        era_lo[idx[:half]] = 1.0
        era_hi[idx[half:]] = 1.0
    out["era_lo"], out["era_hi"] = era_lo, era_hi
    return out


ETT_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load_ettm2(data_dir: str | None = None, holdout: float = 0.2) -> dict:
    """Electricity-transformer temperature (ETTm2): real grid physics.

    Seven channels (6 load levels + oil temperature OT) sampled every 15 min.
    Same identification contract as the fleet loader: z-scored channels, an OU
    fit on within-segment first differences (one segment = consecutive rows of
    the same day inside the same load regime), and a *real* regime shift --
    the high-load era (hours 08-20) vs the low-load era (hours 21-07) -- which
    plays the role the aging drift plays for the fleet.

    The last `holdout` fraction of the timeline is returned separately as
    X_hold/unit_hold for real held-out world-model scoring.
    """
    import csv

    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(here), "data", "ett", "ETTm2.csv")
    rows = []
    with open(data_dir or default, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError("empty ETTm2 csv")

    date_key = next(k for k in rows[0] if k.lower().startswith("date"))
    ot_key = next(k for k in rows[0] if k.strip() == "OT" or k.strip().lower() == "ot")
    cols = [c for c in rows[0] if c not in (date_key,)]
    # keep the canonical 7 channels; map OT case-insensitively
    def _val(row, name):
        if name == "OT":
            return float(row[ot_key])
        return float(row[name])

    X = torch.tensor([[_val(r, c) for c in ETT_COLS] for r in rows], dtype=torch.float64)
    if not torch.isfinite(X).all():
        raise ValueError("non-finite values in ETTm2")
    n_hold = int(X.shape[0] * holdout)
    Xtr = X if n_hold == 0 else X[:-n_hold]
    Xhold = X if n_hold == 0 else X[-n_hold:]
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    Xz = (X - mu) / sd

    # era mask by hour-of-day from the date field ("2016-07-01 00:15:00")
    hours = torch.tensor([int(r[date_key][11:13]) for r in rows], dtype=torch.int64)
    hi_load = (hours >= 8) & (hours < 20)
    # segment ids: (day, era) so first differences never cross a regime boundary
    days = torch.tensor([int(r[date_key][8:10]) + 31 * int(r[date_key][5:7])
                         + 372 * int(r[date_key][:4]) for r in rows], dtype=torch.int64)
    seg = days * 2 + hi_load.to(torch.int64)

    return {"X": Xz, "unit": seg, "X_hold": (Xhold - mu) / sd,
            "unit_hold": seg if n_hold == 0 else seg[-n_hold:], "cols": list(ETT_COLS),
            "era_hi": hi_load, "era_lo": ~hi_load,
            "stats": {"mu": mu.tolist(), "sd": sd.tolist()}}


def era_systems(data: dict) -> tuple[SensorSystem, SensorSystem]:
    """Fit the healthy (early-life) and aged (late-life) era systems."""
    sys_healthy = SensorSystem.fit(data["X"][data["era_lo"] == 1.0],
                                   data["unit"][data["era_lo"] == 1.0])
    sys_aged = SensorSystem.fit(data["X"][data["era_hi"] == 1.0],
                                data["unit"][data["era_hi"] == 1.0])
    return sys_healthy, sys_aged


# ---------------------------------------------------------------------------
# Second real domain: Beijing multi-site air quality (PRSA, 12 stations,
# hourly, 2013-03-01 .. 2017-02-28). Environmental physics: pollutant and
# meteorological channels interact through transport, chemistry and boundary-
# layer dynamics; the seasonal heating cycle is a documented, physically
# grounded regime shift -- the drift event for the gate.
# ---------------------------------------------------------------------------

AQ_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]


def load_prsa_aq(data_dir: str | None = None, station: str = "Aotizhongxin") -> dict:
    """Load one PRSA station and return z-scored hourly channels + season flags.

    Missing values (the CSVs use 'NA') are linearly interpolated in time per
    channel -- standard for this dataset; the filled count is returned so the
    preprocessing is auditable. Channels are z-scored with statistics from the
    *non-heating* era only, so standardization carries no information about the
    drift era. era_lo marks the non-heating baseline regime (certified plant),
    era_hi the heating-season regime (drift plant).
    """
    import csv

    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(os.path.dirname(here), "data", "aqi")
    default_name = f"PRSA_Data_{station}_20130301-20170228.csv"
    cols = ["year", "month", "day", "hour"] + AQ_COLS
    raw: dict[str, list] = {c: [] for c in cols}
    csv_path = data_dir
    if csv_path is None or os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path or default_dir, default_name)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for c in cols:
                v = row[c]
                raw[c].append(float(v) if v not in ("", "NA") else None)
    n = len(raw["year"])
    # linear interpolation over missing values, per channel
    filled = 0
    Xcols = []
    for c in AQ_COLS:
        s = raw[c]
        idx = [i for i, v in enumerate(s) if v is not None]
        out = [None] * n
        for i, v in zip(idx, (s[i] for i in idx)):
            out[i] = v
        filled += n - len(idx)
        if not idx:
            raise ValueError(f"channel {c} entirely missing")
        # edges: extend the nearest valid value; interior: linear in time
        for i in range(0, idx[0]):
            out[i] = s[idx[0]]
        for i in range(idx[-1] + 1, n):
            out[i] = s[idx[-1]]
        for (a, b) in zip(idx, idx[1:]):
            if b - a > 1:
                va, vb = s[a], s[b]
                for k in range(1, b - a):
                    out[a + k] = va + (vb - va) * k / (b - a)
        Xcols.append(out)
    X = torch.tensor(Xcols, dtype=torch.float64).T            # (N, D)
    month = torch.tensor(raw["month"])
    day = torch.tensor(raw["day"])
    # Beijing heating season: Nov 15 - Mar 15 (the municipal schedule)
    heating = ((month == 11) & (day >= 15)) | (month == 12) | (month == 1) | \
              ((month == 3) & (day <= 15))
    era_hi = heating.to(torch.float64)
    era_lo = 1.0 - era_hi
    # standardize on the baseline (non-heating) era only
    mu = X[era_lo == 1.0].mean(dim=0)
    sd = X[era_lo == 1.0].std(dim=0).clamp_min(1e-9)
    Xz = (X - mu) / sd
    return {"X": Xz, "era_lo": era_lo, "era_hi": era_hi, "cols": list(AQ_COLS),
            "station": station, "n_rows": n, "n_filled": filled,
            "frac_heating": float(era_hi.mean()), "mu": mu, "sd": sd}


# ---------------------------------------------------------------------------
# Fourth real domain: continuous stirred-tank reactor sensor array (Kaggle,
# eddardd/continuous-stirred-tank-reactor-domain-adaptation). Chemical-reactor
# physics at plant scale: 1404 sensor channels, 2860 consecutive samples with
# lag-1 autocorrelation ~0.95, and a *real* regime change -- the final quarter
# of the timeline both shifts the attractor (~0.3 sd per channel) and triples
# its variance, the runaway-adjacent behaviour reactor safety cares about.
# Channels are subsampled by an explicit stride so the identified dimension D
# is a knob: stride 8 gives D = 175 -- the first real domain above D = 100.
# ---------------------------------------------------------------------------

CSTR_STRIDE = 8


def load_cstr(data_dir: str | None = None, stride: int = CSTR_STRIDE) -> dict:
    """Load the CSTR sensor array, subsample channels, z-score on the calm era.

    Returns the same contract as the other real loaders: X (N, D) z-scored,
    era_lo / era_hi flags (era_hi = last quarter of the timeline, the disturbed
    regime), unit = segment ids (one continuous segment per era half, so first
    differences used by the OU fit never cross the regime boundary), plus the
    held-out last 15% of the timeline for real world-model scoring.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(here), "data", "cstr",
                           "cstr_rawdata.npy")
    path = data_dir or default
    raw = torch.tensor(np.load(path), dtype=torch.float64)      # (N, 1404)
    if raw.shape[0] < 500 or raw.shape[1] < 100:
        raise ValueError(f"unexpected CSTR array shape {tuple(raw.shape)}")
    if not torch.isfinite(raw).all():
        raise ValueError("non-finite values in CSTR array")
    X = raw[:, ::stride]
    N = X.shape[0]
    # eras: quarters of the timeline; era_hi = final quarter (disturbed regime)
    q = N // 4
    era_hi = torch.zeros(N, dtype=torch.float64)
    era_hi[3 * q:] = 1.0
    era_lo = 1.0 - era_hi
    # standardize on the calm era only
    mu = X[era_lo == 1.0].mean(dim=0)
    sd = X[era_lo == 1.0].std(dim=0).clamp_min(1e-9)
    Xz = (X - mu) / sd
    # segment ids: two halves within each era so the OU fit's first differences
    # never cross the regime boundary (era boundary is the regime switch)
    seg = torch.arange(N, dtype=torch.int64)
    seg = seg // (N // 2)
    seg = seg * 2 + (era_hi == 1.0).to(torch.int64)
    n_hold = int(N * 0.15)
    return {"X": Xz, "era_lo": era_lo, "era_hi": era_hi, "unit": seg,
            "X_hold": Xz[-n_hold:], "unit_hold": seg[-n_hold:],
            "cols": [f"s{j:04d}" for j in range(0, raw.shape[1], stride)],
            "n_rows": N, "D": X.shape[1], "stride": stride, "mu": mu, "sd": sd}


def load_sarcos(data_dir: str | None = None, holdout: float = 0.15) -> dict:
    """SARCOS 7-DoF robot arm (44484 samples, 21 joint inputs + 7 torques).

    Consecutive rows are a smooth measured trajectory (verified: consecutive-row
    deltas are ~23x smaller than shuffled). The plant state is the 7 torques
    (the actuated dynamics the robot actually applies); the 21 kinematic inputs
    provide the exogenous conditioning used only for era split sanity. The
    trajectory is standardized and split: first half = nominal era, final
    quarter = disturbed era (the SARCOS recording contains fast motion phases,
    which act as the regime change for the gate test).
    """
    import scipy.io as sio
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(here), "data", "sarcos",
                           "sarcos_inv.mat")
    path = data_dir or default
    mat = sio.loadmat(path)
    key = [k for k in mat if not k.startswith("__")][0]
    raw = np.asarray(mat[key], dtype=np.float64)                  # (N, 28)
    if raw.shape[1] != 28 or raw.shape[0] < 10000:
        raise ValueError(f"unexpected SARCOS shape {raw.shape}")
    X = torch.tensor(raw[:, 21:], dtype=torch.float64)            # torques (N, 7)
    kin = torch.tensor(raw[:, :21], dtype=torch.float64)
    if not torch.isfinite(X).all():
        raise ValueError("non-finite values in SARCOS torques")
    N = X.shape[0]
    mu, sd = X.mean(dim=0), X.std(dim=0).clamp_min(1e-9)
    Xz = (X - mu) / sd
    # eras: nominal = first 60%, disturbed = final 25% (fast-motion phases),
    # a 15% guard band between them so first differences never cross eras.
    era_hi = torch.zeros(N, dtype=torch.float64)
    era_hi[int(0.75 * N):] = 1.0
    era_lo = torch.zeros(N, dtype=torch.float64)
    era_lo[:int(0.60 * N)] = 1.0
    # segments: 512-row windows, with new ids forced at the era boundaries so
    # no window straddles a regime switch (the OU fit's first differences stay
    # within-era by construction).
    idx = torch.arange(N, dtype=torch.int64)
    seg = idx // 512
    seg = seg + (idx >= int(0.60 * N)).to(torch.int64) \
        + (idx >= int(0.75 * N)).to(torch.int64)
    # kinematic conditioning variance per era (the regime-shift statistic)
    kin_sd_lo = float(kin[era_lo == 1.0].std(dim=0).mean())
    kin_sd_hi = float(kin[era_hi == 1.0].std(dim=0).mean())
    n_hold = int(N * holdout)
    return {"X": Xz, "era_lo": era_lo, "era_hi": era_hi, "unit": seg,
            "X_hold": Xz[-n_hold:], "unit_hold": seg[-n_hold:],
            "kin_sd_ratio": kin_sd_hi / max(kin_sd_lo, 1e-9),
            "cols": [f"tau{j}" for j in range(7)],
            "n_rows": N, "D": 7, "mu": mu, "sd": sd}
