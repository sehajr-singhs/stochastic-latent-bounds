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

import torch

# 8 operational channels of FD001 (1-indexed as in the C-MAPSS schema):
# T24, T30, T50, P30, Ps30, phi, NRf, BPR -- the standard informative subset.
SENSOR_COLS = [2, 3, 4, 8, 9, 15, 13, 12]  # 1-indexed within the 3+21 block
SENSOR_NAMES = ["T24", "T30", "T50", "P30", "Ps30", "phi", "NRf", "BPR"]


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
        sqrt(diag). Used only for random initialisation and region sizing."""
        D = self.dim
        S = torch.diag_embed(self.sigma_vec)
        # vec convention: vec(A X B) = (B^T kron A) vec(X); here A X + X A^T
        K = torch.kron(self.A, torch.eye(D, dtype=self.A.dtype)) + \
            torch.kron(torch.eye(D, dtype=self.A.dtype), self.A)
        sol = torch.linalg.solve(K, (-S.reshape(-1, 1)))
        Sigma_stat = sol.reshape(D, D)
        Sigma_stat = 0.5 * (Sigma_stat + Sigma_stat.T)
        # numerical safety: PD-ify via eigenvalue floor
        w, Vv = torch.linalg.eigh(Sigma_stat)
        Sigma_stat = Vv @ torch.diag(w.clamp_min(1e-8)) @ Vv.T
        return torch.sqrt(torch.diagonal(Sigma_stat)).clamp_min(1e-4)

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
    mu = Xfit.mean(dim=0)
    sd = Xfit.std(dim=0).clamp_min(1e-9)
    Xz = (X - mu) / sd
    out = {"X": Xz[fit_mask], "unit": unit[fit_mask],
           "X_hold": Xz[~fit_mask], "unit_hold": unit[~fit_mask],
           "mu": mu, "sd": sd,
           "sensors": sensors, "names": [SENSOR_NAMES[s - 1] if s - 1 < len(SENSOR_NAMES)
                                         else f"s{s}" for s in sensors]}
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


def era_systems(data: dict) -> tuple[SensorSystem, SensorSystem]:
    """Fit the healthy (early-life) and aged (late-life) era systems."""
    sys_healthy = SensorSystem.fit(data["X"][data["era_lo"] == 1.0],
                                   data["unit"][data["era_lo"] == 1.0])
    sys_aged = SensorSystem.fit(data["X"][data["era_hi"] == 1.0],
                                data["unit"][data["era_hi"] == 1.0])
    return sys_healthy, sys_aged
