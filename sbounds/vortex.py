"""Karman vortex street as a stochastic fluid system (random vortex method).

The fluid-physics domain of the campaign. In two dimensions the incompressible
Euler equations are *exactly* a point-vortex dynamical system, and Chorin's
random vortex method turns Navier-Stokes into the same system plus a Brownian
core walk: the vorticity is carried by discrete vortices whose positions x_i
satisfy the SDE

    dx_i = [ V_ind(x_i) - V_ind(X*) - lambda_flush (x_i - X*_i) ] dt
           + sqrt(2 nu) dW_i ,   i = 1..n,

where V_ind is the Biot-Savart velocity induced by all vortices (Lamb-Oseen
cores of radius a regularise the 1/r singularity; each vortex excludes its own
induction), X* is the staggered Karman street configuration, and sqrt(2 nu) dW
is Chorin's modelling of viscous diffusion as a random walk of the vorticity
carriers. The drift is stated in the co-moving frame (subtract the street's
self-induced translation V_ind(X*)), so the steady street is an exact
equilibrium: drift(X*) = 0. The -lambda_flush (x - X*) term is the observation-
window closure: in a frame tracking the street, perturbation vorticity is
advected downstream and leaves the window at rate lambda_flush (the standard
outflow/shifted-periodic treatment of localized vortex patches; a finite
street's edge modes destabilize otherwise, so some window relaxation is
physically mandatory, not cosmetic). Same-row spacing h, cross-row stagger
h/2, row offset b = 0.281 h -- the classical stability optimum.

This is a genuinely nonlinear plant: the Biot-Savart kernel is 1/r^2 in the
separations, so the drift is not linearisable away -- the coordinate map has
real curvature to undo. State x in R^D with D = 2 n_v, interleaved
(x1, y1, x2, y2, ...) exactly as X_star.reshape(-1).

Interface identical to ChainArm / SensorSystem: dim, drift, diffusion,
equilibrium, lqr_like_scale, noise == "diagonal", float64 throughout.
"""
from __future__ import annotations

import torch


class VortexStreet:
    """Stochastic regularised point-vortex street (random vortex method)."""

    noise = "diagonal"

    def __init__(self, n_vortices: int = 50, U: float = 1.0,
                 gamma0: float = 1.0, spacing: float = 0.5,
                 core_radius: float = 0.08, nu: float = 2e-4,
                 row_offset: float = 0.281, flush: float = 0.5,
                 seed: int = 0):
        self.n_v = n_vortices
        self.U = U                    # mean stream (advection scale)
        self.gamma0 = gamma0          # circulation magnitude per vortex
        self.h = spacing              # same-row streamwise spacing
        self.a = core_radius          # Lamb-Oseen core radius
        self.nu = nu                  # viscous diffusivity (core-walk rate)
        self.row_offset = row_offset  # b/h: 0.281 neutral (Karman), larger = unstable
        self.flush = flush            # window flushing rate (outflow closure)
        D = 2 * n_vortices
        self.dim = D
        g = torch.Generator().manual_seed(seed)

        # staggered Karman street: upper row (even k) at x = k h, y = +b/2 with
        # circulation +gamma0; lower row (odd k) at x = (k+1/2) h, y = -b/2 with
        # circulation -gamma0. b/h = 0.281 is the classical neutral stability
        # offset (Karman); larger offsets give a linearly unstable street -- the
        # physical drift knob for the gate experiment.
        b = row_offset * spacing
        k = torch.arange(n_vortices, dtype=torch.float64)
        upper = (torch.arange(n_vortices) % 2 == 0).to(torch.float64)
        xs = k * spacing + (1.0 - upper) * (0.5 * spacing)
        ys = upper * (0.5 * b) + (1.0 - upper) * (-0.5 * b)
        X_star = torch.stack([xs, ys], dim=1)                  # (n, 2)
        self.X_star = X_star
        self.x_star = X_star.reshape(-1).clone()               # (D,)
        signs = torch.where(upper > 0.5, 1.0, -1.0)            # (n,)
        self._gamma = gamma0 * signs

        # Biot-Savart kernel with Lamb-Oseen core, evaluated at a configuration:
        #   v_i = sum_{j != i} gamma_j * perp(r_ij) / (2 pi |r_ij|^2) * (1 - e^{-r^2/a^2})
        # implemented as a function of x so the drift is exact at any state.
        # Base separations (for the equilibrium drift):
        self._eye = torch.eye(n_vortices, dtype=torch.float64)
        # Constant equilibrium induction, computed once (2x drift speedup).
        self._v_star = self._induced(self.X_star.unsqueeze(0))[0].reshape(-1)

    # --- kernel ---------------------------------------------------------------
    def _induced(self, pos: torch.Tensor) -> torch.Tensor:
        """Induced velocity at each vortex center: pos (..., n, 2) -> (..., n, 2)."""
        diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)           # (..., i, j, 2): x_i - x_j
        r2 = (diff ** 2).sum(dim=-1).clamp_min(1e-12)          # (..., i, j)
        perp = torch.stack([-diff[..., 1], diff[..., 0]], dim=-1)
        core = 1.0 - torch.exp(-r2 / self.a ** 2)
        vel_j = perp / (2 * torch.pi * r2.unsqueeze(-1)) * core.unsqueeze(-1)
        vel_j = vel_j * (1.0 - self._eye).unsqueeze(-1)        # no self-induction
        return (vel_j * self._gamma.view(1, -1, 1)).sum(dim=-2)  # (..., n, 2)

    # --- the System interface ------------------------------------------------
    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """Co-moving-frame drift: V_ind(x) - V_ind(X*).

        Subtracting the street's self-induced translation makes the steady
        street an exact equilibrium of the closed system; the mean stream U
        advects the whole frame and cancels identically in relative dynamics.
        """
        lead = x.shape[:-1]
        pos = x.unflatten(-1, (self.n_v, 2))
        v = (self._induced(pos).flatten(-2) - self._v_star) \
            - self.flush * (x - self.x_star)
        return v

    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        # Chorin core walk: sqrt(2 nu) per coordinate, isotropic, constant.
        # Convention of the other plants: batched (..., D, D).
        s = (2.0 * self.nu) ** 0.5
        lead = x.shape[:-1]
        eye = torch.eye(self.dim, dtype=x.dtype).expand(lead + (self.dim, self.dim))
        return eye * s

    def equilibrium(self) -> torch.Tensor:
        return self.x_star

    def lqr_like_scale(self) -> torch.Tensor:
        """Sizing only (never enters a sound bound): the street is Hamiltonian
        (neutrally stable), so there is no OU stationary spread; perturbations
        are sized as a fixed fraction of the spacing h per coordinate."""
        return torch.full((self.dim,), 0.10 * self.h, dtype=torch.float64)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"VortexStreet(n_v={self.n_v}, D={self.dim}, U={self.U}, "
                f"a={self.a}, nu={self.nu}, b/h={self.row_offset})")
