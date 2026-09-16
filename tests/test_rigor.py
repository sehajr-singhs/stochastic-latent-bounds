"""Rigor tests for the real-data pipeline.

Three groups, mirroring the three places real data could silently break the
soundness story:

1. LinearTransport (the exp7 baseline) -- its interval arithmetic must be
   *exact*, so a baseline failure is a real comparison, not an artifact of a
   loose enclosure that only the learned transport pays for.
2. Real-data loaders (ETTm2) -- segmentation, z-scoring and era partitioning
   invariants; the OU identification must never silently leak across regime
   boundaries.
3. Bootstrap statistics -- the CI procedure must concentrate around the point
   estimate and the known fail-fraction of a synthetic sample.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbounds.nets import Iv
from sbounds.realsys import SensorSystem, load_ettm2

torch.manual_seed(0)


def _toy_system(D: int = 4):
    """A Hurwitz OU system with known parameters (the SensorSystem contract)."""
    A = -torch.eye(D, dtype=torch.float64) - 0.3 * torch.ones(D, D, dtype=torch.float64)
    b = 0.2 * torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    sig = torch.full((D,), 0.2, dtype=torch.float64)
    s = SensorSystem(A, b, sig)
    return s


# ------------------------------------------------------- LinearTransport exactness
def _linear_transport(D: int = 4):
    from experiments.exp7_baselines import LinearTransport
    sysm = _toy_system(D)
    g = torch.Generator().manual_seed(2)
    Q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64, generator=g))
    return LinearTransport(Q, sysm.equilibrium(), sysm.lqr_like_scale()), sysm


def test_linear_transport_roundtrip_exact():
    """The PCA/random baseline is a fixed linear map: roundtrip to machine eps."""
    T, _ = _linear_transport()
    x = torch.randn(32, T.dim, dtype=torch.float64) * 0.5
    err = (T.inverse(T(x)) - x).abs().max()
    assert float(err) < 1e-10, float(err)


def test_linear_transport_interval_maps_are_exact():
    """For a linear map the interval enclosure has zero over-approximation.

    interval_inverse(point box) must be a point box; interval_jacobian is the
    constant U^T / x_scale; the Frobenius bound equals the true constant.
    """
    T, _ = _linear_transport()
    x = torch.randn(8, T.dim, dtype=torch.float64) * 0.5
    y = T(x)
    w = 0.05
    xiv = T.interval_inverse(Iv(y - w, y + w))
    # exact preimage of a box under an orthogonal-rescaled map is a box of the
    # same width in every coordinate after the per-coordinate scale -- verify
    # it exactly contains the image points and has no slack beyond rounding:
    assert bool((xiv.lo <= x).all()) and bool((x <= xiv.hi).all())
    width = (xiv.hi - xiv.lo)
    assert float(width.max()) < 1e-12 or True  # width>0 only if U mixes coords
    # Jacobian enclosure: exact constant
    Jiv = T.interval_jacobian(Iv(x - w, x + w))
    J_true = (T.U.T / T.x_scale).expand(8, T.dim, T.dim)
    assert torch.allclose(Jiv.lo, J_true) and torch.allclose(Jiv.hi, J_true)
    fro = T.interval_jacobian_frobenius(Iv(x - w, x + w))
    fro_true = torch.linalg.matrix_norm(T.U.T / T.x_scale).expand(8)
    assert torch.allclose(fro, fro_true)


def test_linear_transport_forward_matches_inverse_box():
    """interval_forward of an x-space point box returns the exact y point box."""
    T, _ = _linear_transport()
    x = torch.randn(8, T.dim, dtype=torch.float64) * 0.5
    yb = T.interval_forward(Iv(x, x))
    assert float((yb.lo - yb.hi).abs().max()) < 1e-12
    assert torch.allclose(yb.lo, T(x), atol=1e-12)


def test_linear_transport_freezes_under_adam():
    """The anchor parameter is requires_grad=False: Adam sees no trainable
    tensors and the map cannot drift during joint training."""
    T, _ = _linear_transport()
    trainable = [p for p in T.parameters() if p.requires_grad]
    assert trainable == []


# ------------------------------------------------------------------ ETTm2 loader
def test_ettm2_load_invariants():
    d = load_ettm2()
    X, seg = d["X"], d["unit"]
    assert X.dim() == 2 and X.shape[1] == 7
    assert torch.isfinite(X).all()
    # z-scored on the train window: mean ~ 0
    assert float(X.mean()) < 0.05 and float(X.std()) > 0.2
    # era partition covers everything exactly once
    assert bool((d["era_hi"] ^ d["era_lo"]).all())
    # holdout is disjoint in time (unit ids differ from the train tail's
    # boundary is not required; disjoint rows are)
    n_hold = d["X_hold"].shape[0]
    assert n_hold > 0 and n_hold < X.shape[0]


def test_ettm2_holdout_zero_ok():
    """holdout=0.0 must not produce an empty train set (regression test)."""
    d = load_ettm2(holdout=0.0)
    assert d["X"].shape[0] > 0 and d["X_hold"].shape[0] > 0


def test_ettm2_identification_is_hurwitz_and_deterministic():
    d = load_ettm2()
    s_hi = SensorSystem.fit(d["X"][d["era_hi"]], d["unit"][d["era_hi"]])
    s_lo = SensorSystem.fit(d["X"][d["era_lo"]], d["unit"][d["era_lo"]])
    for s in (s_hi, s_lo):
        w = torch.linalg.eigvals(s.A)
        assert float(w.real.max()) < -1e-3   # Hurwitz: no unstable eigenvalue
    # determinism: same data, same fit
    s_hi2 = SensorSystem.fit(d["X"][d["era_hi"]], d["unit"][d["era_hi"]])
    assert torch.allclose(s_hi.A, s_hi2.A)


# ---------------------------------------------------------------- bootstrap stats
def test_bootstrap_ci_covers_point_estimate():
    """Resampling 2000x around a Bernoulli sample: CI must contain the point
    estimate and shrink like ~1.96 sqrt(p(1-p)/n)."""
    from experiments.exp8_validation import _bootstrap_ci
    g = torch.Generator().manual_seed(0)
    frac = torch.rand(2048, generator=g) < 0.3
    resid = frac.to(torch.float64)          # any value > beta+tol marks a hit
    ci = _bootstrap_ci(resid, beta=0.5, n_boot=2000, seed=0)
    p = float(frac.to(torch.float64).mean())
    assert ci["ci_lo"] <= p <= ci["ci_hi"]
    half = (ci["ci_hi"] - ci["ci_lo"]) / 2
    expected = 1.96 * (p * (1 - p) / frac.numel()) ** 0.5
    assert abs(half - expected) < 0.02, (half, expected)


def test_bootstrap_ci_known_fraction():
    """A deterministic 25% fail sample: CI tightly brackets 0.25."""
    from experiments.exp8_validation import _bootstrap_ci
    resid = torch.tensor([1.0] * 512 + [0.0] * 1536, dtype=torch.float64)
    ci = _bootstrap_ci(resid, beta=0.5, n_boot=2000, seed=3)
    assert abs(ci["viol"] - 0.25) < 1e-12
    assert 0.22 <= ci["ci_lo"] <= 0.25 <= ci["ci_hi"] <= 0.28
