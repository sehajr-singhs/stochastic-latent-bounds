"""Learned certificate components: the latent Lyapunov function V(eta) and the
latent stochastic dynamics F(y) of the world model.

Both are constructed so the certificate's algebraic requirements hold *by
construction* rather than by hoping the optimizer finds them:

  V(eta) = 1/2 eta^T P eta + sum_j softplus(c_j) * n_j(eta)^2,   P = A A^T + eps I
      * P is positive definite by construction (no eigenvalue search)
      * n_j(eta) = res_j(eta) - res_j(0), so n_j(0) = 0 and the residual cannot
        make V(0) nonzero
      * the coefficient softplus(c_j) >= 0 guarantees V >= 1/2 eta^T P eta, so V
        is positive definite and radially unbounded with no extra penalty term

  F(y) = A y + (res(y) - res(0)),  so F(0) = 0 exactly
      * the equilibrium of the physical system maps to y = 0, so the learned
        latent drift having an exact zero is a real constraint, not a nicety
"""
from __future__ import annotations

import torch

from .nets import Iv, MLP, _mul_iv, _round, iv_matmul, spectral_norm_bound


def _mm_exact_batched(Jlo: torch.Tensor, Jhi: torch.Tensor, dev: Iv) -> Iv:
    """Sound enclosure of J @ dev for interval J (B, O, D) and interval dev (B, D).

    Per-coefficient corner products contain the true products; summing the
    per-coefficient minima (maxima) stays below (above) the true row sums, so
    the result is a sound enclosure of the batched matvec. Used for the
    mean-value remainder terms.
    """
    dl = dev.lo.unsqueeze(-2)                      # (B, 1, D)
    dh = dev.hi.unsqueeze(-2)                      # (B, 1, D)
    corners = torch.stack([Jlo * dl, Jlo * dh, Jhi * dl, Jhi * dh], dim=0)
    lo = corners.min(dim=0).values.sum(-1)         # (B, O)
    hi = corners.max(dim=0).values.sum(-1)
    return Iv(lo, hi)


class LyapunovNet(torch.nn.Module):
    """Positive-definite-by-construction Lyapunov function on the latent factor."""

    def __init__(self, d: int, width: int = 32, depth: int = 2, n_res: int = 16,
                 act: str = "tanh", eps: float = 1e-3, seed: int = 0):
        super().__init__()
        self.d = d
        self.eps = float(eps)
        self.A = torch.nn.Parameter(torch.eye(d, dtype=torch.float64) * 0.5)
        self.c = torch.nn.Parameter(torch.full((n_res,), -2.0, dtype=torch.float64))
        self.res = MLP(d, n_res, width, depth, act, "linear", seed=seed)

    @property
    def P(self) -> torch.Tensor:
        return self.A @ self.A.T + self.eps * torch.eye(self.d, dtype=self.A.dtype)

    def _eta0(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.d, dtype=ref.dtype, device=ref.device)

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        P = self.P
        n = self.res(eta) - self.res(self._eta0(eta))
        quad = 0.5 * ((eta @ P) * eta).sum(-1)
        return quad + (torch.nn.functional.softplus(self.c) * n ** 2).sum(-1)

    def value_at_origin(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ref.shape[:-1], dtype=ref.dtype)

    # --- sound bounds -------------------------------------------------
    @torch.no_grad()
    def bounds(self, lo: torch.Tensor, hi: torch.Tensor):
        """Sound enclosures of V, grad V and Hess V over the eta-box [lo, hi].

        Returns (V Iv[(B,)], grad Iv[(B,d)], hess Iv[(B,d,d)]).
        """
        B, d = lo.shape
        res_val, res_jac, res_hess = self.res.jet(lo, hi)      # (B,R), (B,R,d), (B,R,d,d)
        res0 = self.res(torch.zeros(d, dtype=lo.dtype)).detach()
        n_val = _round(res_val - res0)                          # Iv (B,R)
        a = torch.nn.functional.softplus(self.c).detach()       # (R,)
        eta = Iv(lo, hi)
        P = self.P.detach()
        Peta = iv_matmul(P, eta)
        # elementwise product then reduce, explicitly per bound
        prod = eta * Peta
        quad = Iv((prod.lo).sum(-1), (prod.hi).sum(-1))
        n_sq = n_val.sq()
        V = _round(0.5 * quad + Iv((n_sq.lo * a).sum(-1), (n_sq.hi * a).sum(-1)))

        grad = iv_matmul(P, eta)
        for j in range(self.res.n_out):
            nj = n_val[:, j:j + 1]
            gj = res_jac[:, j, :]
            grad = grad + _round((2.0 * a[j]) * (nj * gj))
        P_b = P.expand(B, d, d).contiguous()
        hess = Iv(P_b.clone(), P_b.clone())
        for j in range(self.res.n_out):
            gj = res_jac[:, j, :]
            outer = _mul_iv(gj.unsqueeze(-1), gj.unsqueeze(-2))
            hj = res_hess[:, j, :, :]
            hess = hess + _round((2.0 * a[j]) * (outer + n_val[:, j:j + 1].unsqueeze(-1) * hj))
        return V, grad, hess

    @torch.no_grad()
    def hessian_frobenius(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Sound upper bound on ||Hess V||_F over the box."""
        _, _, hess = self.bounds(lo, hi)
        return torch.sqrt((hess.absmax ** 2).sum(dim=(-1, -2)).clamp_min(0.0))


class LatentDynamics(torch.nn.Module):
    """Latent stochastic dynamics F(y) with F(0) = 0 enforced structurally."""

    def __init__(self, dim: int, width: int = 64, depth: int = 2, act: str = "tanh",
                 seed: int = 0, A_init: torch.Tensor | None = None,
                 d_eta: int | None = None):
        super().__init__()
        self.dim = dim
        A0 = torch.zeros(dim, dim, dtype=torch.float64) if A_init is None else A_init.clone()
        self.A = torch.nn.Parameter(A0)
        self.d_eta = d_eta
        if d_eta is None:
            self.res = MLP(dim, dim, width, depth, act, "linear", seed=seed)
        else:
            # Factorised residual: the certified factor's rows read ONLY the
            # factor coordinates. The eta-rows of the residual are then exactly
            # rho-independent, which is the structural property the sound
            # centered bounds need (a Jacobian enclosure with a zero block
            # instead of a Lipschitz ball over the transversal coordinates).
            self.res_eta = MLP(d_eta, d_eta, max(width // 2, 16), depth, act, "linear", seed=seed)
            self.res_rho = MLP(dim, dim - d_eta, width, depth, act, "linear", seed=seed + 1)

    def _residual(self, y: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """res(y) - res(y_ref), zero at y = y_ref by construction."""
        d = self.dim
        if self.d_eta is None:
            return self.res(y) - self.res(y_ref)
        de = self.res_eta(y[..., :self.d_eta]) - self.res_eta(y_ref[..., :self.d_eta])
        dr = self.res_rho(y) - self.res_rho(y_ref)
        return torch.cat([de, dr], dim=-1)

    def residual(self, y: torch.Tensor, y_ref: torch.Tensor | None = None) -> torch.Tensor:
        if y_ref is None:
            y_ref = torch.zeros(self.dim, dtype=y.dtype, device=y.device)
        return self._residual(y, y_ref)

    def residual_rho(self, y: torch.Tensor, y_ref: torch.Tensor | None = None,
                     d_eta: int | None = None) -> torch.Tensor:
        """The rho rows of res(y) - res(y_ref)."""
        d = d_eta if d_eta is not None else (self.d_eta or 0)
        return self.residual(y, y_ref)[..., d:]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(self.dim, dtype=y.dtype, device=y.device)
        return y @ self.A.T + self._residual(y, zero)

    # --- sound bounds -------------------------------------------------
    @torch.no_grad()
    def ibp_box(self, lo: torch.Tensor, hi: torch.Tensor) -> Iv:
        """Sound enclosure of F over the y-box, in *centered* (mean-value) form.

        The raw IBP (interval bound propagation through the residual MLP) is
        sound but its width compounds through the layers and the resulting
        drift enclosure is far too loose for branch-and-bound to certify
        anything at region scale. The centered form evaluates the residual
        exactly at the box center (autograd) and bounds the deviation by the
        mean-value theorem with a *Jacobian enclosure*:

            res(y) in res(c) + Jbox @ (y - c),

        where Jbox row-blocks are exact zeros or Lipschitz balls, exploiting
        the factorised residual structure (eta rows read only eta). The linear
        part A y is enclosed tightly by interval arithmetic. The result is
        sound and O(box radius) instead of exponentially loose.
        """
        d = self.dim
        zero = torch.zeros(d, dtype=lo.dtype)
        y_iv = Iv(lo, hi)
        lin = iv_matmul(self.A.detach(), y_iv)
        c = 0.5 * (lo + hi)
        yc = c.clone().requires_grad_(True)
        res_c = self._residual(yc, zero)
        J = torch.stack([torch.autograd.grad(res_c[..., k].sum(), yc,
                                             retain_graph=True)[0] for k in range(d)], dim=-2)
        if self.d_eta is None:
            L = spectral_norm_bound(self.res)
            Jlo = torch.full_like(J, -L)
        else:
            L_eta = spectral_norm_bound(self.res_eta)
            L_rho = spectral_norm_bound(self.res_rho)
            row_e = (torch.arange(d) < self.d_eta).view(1, d, 1)
            col_e = (torch.arange(d) < self.d_eta).view(1, 1, d)
            # eta rows: d res_eta / d rho = 0 exactly; d res_eta / d eta = Lipschitz ball
            Jlo = torch.where(row_e & col_e, torch.full_like(J, -L_eta),
                              torch.where(row_e, torch.zeros_like(J),
                                          torch.full_like(J, -L_rho)))
        Jhi = -Jlo                                   # symmetric enclosure
        dev = Iv(lo - c, hi - c)                     # exact: y - c over the box
        rem = _mm_exact_batched(Jlo, Jhi, dev)
        res_iv = _round(Iv(res_c.detach() + rem.lo, res_c.detach() + rem.hi))
        res0 = self._residual(zero, zero).detach()   # exact zero-centring constant
        return _round(lin + res_iv - res0)

    @torch.no_grad()
    @torch.no_grad()
    def ibp_box_chunked(self, lo: torch.Tensor, hi: torch.Tensor, chunk: int = 512) -> tuple:
        outs_lo, outs_hi = [], []
        for s in range(0, lo.shape[0], chunk):
            iv = self.ibp_box(lo[s:s + chunk], hi[s:s + chunk])
            outs_lo.append(iv.lo)
            outs_hi.append(iv.hi)
        return torch.cat(outs_lo), torch.cat(outs_hi)

    # --- structural read-outs ----------------------------------------
    def blocks(self, d_eta: int):
        A = self.A.detach()
        return (A[:d_eta, :d_eta], A[:d_eta, d_eta:], A[d_eta:, :d_eta], A[d_eta:, d_eta:])

    def rho_contraction_rate(self, d_eta: int) -> torch.Tensor:
        """lambda_max of the symmetric part of A_rho_rho.

        Reported for diagnostics only. It is a *bad* certificate test for a
        mechanical system and the repo says so: the linearisation of a mass-
        spring-damper system in position-velocity coordinates has
        [[0, I],[-M^-1 K, -M^-1 D]], whose symmetric part has a zero diagonal
        block and therefore a non-negative largest eigenvalue no matter how
        damped the system is. Contraction has to be measured through a metric,
        which is what `rho_metric` supplies.
        """
        _, _, _, Arr = self.blocks(d_eta)
        return torch.linalg.eigvalsh(0.5 * (Arr + Arr.T)).max()

    def refresh_rho_metric(self, d_eta: int) -> dict:
        """Solve the Lyapunov equation for the rho block and cache the metric.

        Finds Q > 0 with A_rr^T Q + Q A_rr = -I, so that rho^T Q F_rho carries
        the exact term -||rho||^2. If the learned rho block is not Hurwitz, or Q
        comes out non-positive-definite, the metric falls back to the identity
        and the flag is recorded: the certificate is then simply not available
        for this model, which is a result and not a crash.
        """
        _, _, _, Arr = self.blocks(d_eta)
        Q = solve_lyapunov_metric(Arr)
        finite = bool(torch.isfinite(Q).all())
        lmin = float(torch.linalg.eigvalsh(Q).min()) if finite else float("-inf")
        ok = bool(finite and lmin > 0.0)
        if not ok:
            Q = torch.eye(Arr.shape[0], dtype=Arr.dtype)
        self._Q = Q
        self._q_ok = ok
        self._q_lmin = lmin
        return {"rho_metric_ok": ok, "rho_metric_lmin": lmin,
                "symmetric_part_lmax": float(self.rho_contraction_rate(d_eta))}

    def rho_metric(self, d_eta: int) -> torch.Tensor:
        Q = getattr(self, "_Q", None)
        if Q is None or Q.shape[0] != self.dim - d_eta:
            self.refresh_rho_metric(d_eta)
            Q = self._Q
        return Q


def solve_lyapunov_metric(A: torch.Tensor) -> torch.Tensor:
    """Return the symmetric Q solving A^T Q + Q A = -I, or a NaN sentinel.

    Uses the row-major vectorisation identity vec(AXB) = (A (x) B^T) vec(X), so
    the system is (A^T (x) I + I (x) A^T) vec(Q) = -vec(I).
    """
    A = A.contiguous()
    m = A.shape[0]
    I = torch.eye(m, dtype=A.dtype)
    AT = A.T.contiguous()
    K = torch.kron(AT, I) + torch.kron(I, AT)
    try:
        q = torch.linalg.solve(K, -I.reshape(-1))
    except Exception:
        return torch.full((m, m), float("nan"), dtype=A.dtype)
    Q = q.reshape(m, m)
    return 0.5 * (Q + Q.T)


def spectral_project(module: torch.nn.Module, cap: float, iters: int = 8) -> None:
    """Rescale every Linear weight whose spectral norm exceeds `cap`.

    This is the Lipschitz regularisation the certificate's soundness wants: it
    keeps the learned maps' local slope bounded, which keeps the interval
    enclosures from blowing up. It is applied after each optimizer step.
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            with torch.no_grad():
                W = m.weight
                m_, n_ = W.shape
                v = torch.ones(n_, dtype=W.dtype)
                for _ in range(iters):
                    u = W @ v
                    nu = u.norm()
                    if nu == 0:
                        break
                    u = u / nu
                    v = W.T @ u
                    nv = v.norm()
                    v = v / nv if nv > 0 else v
                sigma = float((W @ v).norm())
                if sigma > cap > 0:
                    m.weight.mul_(cap / sigma)


def parameter_count(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())
