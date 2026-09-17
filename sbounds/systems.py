"""Physical systems used as world-model data sources, with process noise.

`ChainArm` is an N-link torque-driven planar robotic arm (point masses at the
link tips) under gravity-compensated PD control. It is deliberately analytic:
the manipulator equation

    M(theta) theta_dd + c(theta, theta_d) + G(theta) = tau

with M_ij = A_ij l_i l_j cos(theta_i - theta_j), A_ij = sum_{k>=max(i,j)} m_k,
c_i = sum_j A_ij l_i l_j sin(theta_i - theta_j) theta_d_j^2, and
G_i = g A_i l_i sin(theta_i), is derived in closed form in `docs/math.md`.
Choosing tau = G(theta) - Kp theta - Kd theta_d makes x* = 0 an *exact*
equilibrium of the closed loop, so the certificate's target is not approximate.

Why an analytic drift matters: the true Ito push-forward of the physical SDE
through an invertible transport has a closed form we can evaluate with
autograd, so the ``world model'' can be scored against ground truth instead of
being taken on faith. `pushforward_drift` is that ground truth.

State x = [theta (N), omega (N)] in R^D with D = 2N. Noise enters in the
velocity block, which is the physical place actuator and damping uncertainty
acts. Two diffusion models are provided:

  additive         Sigma = sigma * [[0],[I]]        (constant, headline)
  state_dependent  Sigma = sigma * diag(1 + |w|)    (multiplicative sweep)

Both are axis-aligned in velocity coordinates, so the Ito trace reduces to a
sum of second derivatives along the velocity axes. The bounds engine does not
exploit that (it uses a Frobenius bound, sound but looser); `generator_exact`
does, which is what the conservatism metric compares against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

TWO_PI = 2.0 * torch.pi


@dataclass
class ChainArm:
    """N-link planar arm, gravity-compensated PD, velocity-block process noise."""

    n_links: int = 8
    mass: float = 1.0
    length: float = 0.5
    gravity: float = 9.81
    kp: float = 40.0
    kd: float = 8.0
    sigma: float = 0.0
    noise: str = "additive"
    dtype: torch.dtype = torch.float64
    _cache: dict = field(default_factory=dict, repr=False)

    # --- geometry -----------------------------------------------------
    @property
    def dim(self) -> int:
        return 2 * self.n_links

    def _A(self) -> torch.Tensor:
        """A_ij = sum_{k >= max(i,j)} m_k, i.e. mass distal to both joints."""
        n, m = self.n_links, self.mass
        idx = torch.arange(n, dtype=self.dtype)
        k = torch.maximum(idx[:, None], idx[None, :])
        # count of k with k >= max(i,j) among 0..n-1
        return (n - k).to(self.dtype) * m

    def _gravity_vec(self) -> torch.Tensor:
        n = self.n_links
        dist = (n - torch.arange(n, dtype=self.dtype))  # sum_{k>=i} m_k
        return self.gravity * dist * self.mass * self.length

    # --- dynamics -----------------------------------------------------
    def inertia(self, theta: torch.Tensor) -> torch.Tensor:
        """M(theta): (..., N, N)."""
        A = self._A()
        l = self.length
        d = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        return A * l * l * torch.cos(d)

    def coriolis(self, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """c(theta, omega)_i = sum_j A_ij l_i l_j sin(theta_i - theta_j) omega_j^2."""
        A = self._A()
        l = self.length
        d = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        return (A * l * l * torch.sin(d) * omega.unsqueeze(-2) ** 2).sum(dim=-1)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """Closed-loop drift f(x) for dx = f(x) dt + Sigma(x) dW."""
        n = self.n_links
        theta, omega = x[..., :n], x[..., n:]
        rhs = -self.kp * theta - self.kd * omega - self.coriolis(theta, omega)
        M = self.inertia(theta)
        alpha = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        return torch.cat([omega, alpha], dim=-1)

    def diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """Sigma(x): (..., D, D). Velocity-block noise."""
        n, D, s = self.n_links, self.dim, self.sigma
        lead = x.shape[:-1]
        eye = torch.eye(n, dtype=x.dtype).expand(lead + (n, n))
        if self.noise == "additive":
            scale = s * eye
        elif self.noise == "state_dependent":
            scale = torch.diag_embed(s * (1.0 + x[..., n:].abs()))
        elif self.noise == "actuator":
            # torque noise mapped through the inertia: Sigma_w = M^{-1} diag(s)
            M = self.inertia(x[..., :n])
            scale = torch.linalg.solve(M, eye * s)
        else:
            raise ValueError(self.noise)
        top = torch.zeros(lead + (n, n), dtype=x.dtype)
        return torch.cat([top, scale], dim=-2)

    # --- reference quantities ----------------------------------------
    def equilibrium(self) -> torch.Tensor:
        """x* = 0 exactly, by the gravity-compensation construction."""
        return torch.zeros(self.dim, dtype=self.dtype)

    def linearization(self) -> torch.Tensor:
        """Jacobian of the drift at x* = 0 (exact: Coriolis is O(|x|^2))."""
        n = self.n_links
        M0 = self.inertia(torch.zeros(n, dtype=self.dtype))
        A = torch.linalg.solve(M0, -self.kp * torch.eye(n, dtype=self.dtype))
        B = torch.linalg.solve(M0, -self.kd * torch.eye(n, dtype=self.dtype))
        top = torch.cat([torch.zeros(n, n, dtype=self.dtype), torch.eye(n, dtype=self.dtype)], dim=1)
        bottom = torch.cat([A, B], dim=1)
        return torch.cat([top, bottom], dim=0)

    def lqr_like_scale(self) -> torch.Tensor:
        """Diagonal positive-definite weight used to scale coordinates.

        Returns s in R^D such that s_i x_i is O(1) over a representative region;
        used to make a symmetric box comparable across states and dims.
        """
        n = self.n_links
        theta = torch.full((n,), 0.6, dtype=self.dtype)      # rad
        omega = torch.full((n,), 2.0 * self.sigma + 0.6, dtype=self.dtype)
        return torch.cat([theta, omega])


@torch.enable_grad()
def pushforward_drift(system: ChainArm, transport, x: torch.Tensor) -> torch.Tensor:
    """Exact Ito push-forward drift of the physical SDE under y = T(x).

    For dx = f dt + Sigma dW and y = T(x):

        dy = ( J_T f + 1/2 tr(Sigma^T H_T Sigma) ) dt + J_T Sigma dW

    The trace is over the second-derivative tensor of each component of T. This
    is the term most latent-dynamics models silently absorb; the repo verifies
    numerically that a world model trained on pushed-forward data recovers an
    approximation of it, and that dropping it makes the fit worse.
    """
    x = x.clone().requires_grad_(True)
    f = system.drift(x)
    sig = system.diffusion(x)
    J = _batched_jacobian(transport, x)                    # (B, D, D)
    H = _batched_hessian(transport, x)                     # (B, D, D, D)
    SigSig = sig @ sig.transpose(-1, -2)                   # (B, D, D)
    corr = 0.5 * torch.einsum("bkij,bij->bk", H, SigSig)   # (B, D)
    return (J @ f.unsqueeze(-1)).squeeze(-1) + corr


@torch.enable_grad()
def _batched_jacobian(fn, x: torch.Tensor) -> torch.Tensor:
    """J[b, k, i] = d T_k / d x_i, one autograd call per component (D is small)."""
    B, D = x.shape
    outs = []
    for k in range(D):
        g = torch.autograd.grad(fn(x)[..., k].sum(), x, retain_graph=True)[0]
        outs.append(g)
    return torch.stack(outs, dim=-2)                       # (B, D, D)


@torch.enable_grad()
def _batched_hessian(fn, x: torch.Tensor) -> torch.Tensor:
    """H[b, k, i, j] = d2 T_k / d x_i d x_j.

    If T's output does not depend on any trainable parameter (e.g. the fixed
    linear PCA/random-projection baselines), the first backward is a constant
    w.r.t. x and carries no graph: the second derivative is then *exactly*
    zero, which is what we return -- that is the mathematics, not a fallback.
    """
    B, D = x.shape
    rows = []
    for k in range(D):
        g = torch.autograd.grad(fn(x)[..., k].sum(), x, create_graph=True)[0]  # (B, D)
        row = []
        for i in range(D):
            try:
                h = torch.autograd.grad(g[:, i].sum(), x, retain_graph=True)[0]
            except RuntimeError:
                return torch.zeros(B, D, D, D, dtype=x.dtype, device=x.device)
            row.append(h)
        rows.append(torch.stack(row, dim=-2))
    return torch.stack(rows, dim=-3)                       # (B, D, D, D)


@torch.no_grad()
def generator_exact(system: ChainArm, transport, V_fn, x: torch.Tensor, kappa: float) -> torch.Tensor:
    """Exact generator of W = V(eta) + kappa||rho||^2 at physical states x.

    Used only for *evaluation* (tightness of the sound bound, Monte-Carlo
    validation). The certificate itself never has access to this.
    """
    x = x.clone().requires_grad_(True)
    y = transport(x)
    f = system.drift(x)
    sig = system.diffusion(x)
    W = W_of_transport(V_fn, y, kappa)
    gW = torch.autograd.grad(W.sum(), x, create_graph=True)[0]
    drift = (gW * f).sum(-1)
    D = x.shape[-1]
    ito = torch.zeros_like(drift)
    for i in range(D):
        gi = torch.autograd.grad(gW[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        ito = ito + (gi * (sig @ sig.transpose(-1, -2))[:, i, :]).sum(-1)
    return drift + 0.5 * ito


def W_of_transport(V_fn, y: torch.Tensor, kappa: float, d: int | None = None) -> torch.Tensor:
    """W(y) = V(eta) + kappa * ||rho||^2 with eta = y[:d], rho = y[d:]."""
    if d is None:
        d = y.shape[-1] // 2
    eta, rho = y[..., :d], y[..., d:]
    return V_fn(eta) + kappa * (rho ** 2).sum(-1)


def sample_region(system: ChainArm, n: int, scale: float = 1.0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform samples in the symmetric operating box, centred on x*."""
    s = system.lqr_like_scale() * scale
    u = torch.rand((n, system.dim), dtype=system.dtype, generator=generator)
    return (2.0 * u - 1.0) * s
