"""Sound bounds on the stochastic generator L W + alpha W.

The certificate asks for

    L W(y) + alpha W(y) <= 0    for every y in the certified region,

where for dy = F(y) dt + B(y) dW the generator is

    L W(y) = grad W . F(y) + 1/2 tr( B B^T Hess W ),

with W(y) = V(eta) + kappa rho^T Q rho and Q the Lyapunov metric of the rho block
(see `generator`). Two modes are implemented, and crucially they certify *the
same function over the same region*; the only difference is how much of the space
is searched.

  full   : y ranges over a D-dimensional box, everything sound-bounded per box.
           Tighter bounds, but the search is over all D coordinates.
  factor : eta ranges over a d-dimensional box while rho stays inside a ball of
           radius r whose contribution is bounded in closed form. The search is
           over d coordinates only, which is the mechanism that beats the curse
           of dimensionality.

The rho bound in factor mode is the load-bearing step, so it is worth writing
out. Split F_rho = A_rr rho + (A_re eta + g_rho) and let
b_Q = sup ||Q (A_re eta + g_rho)|| over the region. Then

    2 kappa rho^T Q F_rho = kappa ( -||rho||^2 + 2 rho^T Q (A_re eta + g_rho) )
                          <= kappa ( -s^2 + 2 s b_Q )     with s = ||rho|| <= r
                          <= kappa * max_{s in [0, r]} ( -s^2 + 2 s b_Q )
                           = kappa b_Q^2                     if r >= b_Q
                           = kappa ( 2 r b_Q - r^2 )         otherwise

using -s^2 + 2 s b_Q = b_Q^2 - (s - b_Q)^2. Completing the square here is what
makes the bound independent of the radius once the ball is wide enough; the naive
split of the two terms would grow linearly in r and certify nothing.

Two places this is deliberately, visibly loose:

  1. The Ito term uses 1/2 tr(BB^T H) <= 1/2 ||B||_F^2 ||H||_F, sound but
     conservative. `ito_tightness` measures the ratio against the exact trace at
     sampled points so the slack is reported rather than hidden.
  2. In factor mode rho is relaxed to the axis-aligned box [-r, r]^m when the two
     coupling norms are bounded, which over-covers the ball.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .nets import Iv, _mul_iv, _round, frobenius_bound, iv_matmul
from .models import _mm_exact_batched, spectral_norm_bound
from .transport import _iv_matmul_batched


@dataclass
class BoundResult:
    """A sound upper bound on sup_region (L W + alpha W), plus its parts."""

    upper: torch.Tensor          # (B,) total
    drift_eta: torch.Tensor      # (B,) grad_eta V . F_eta
    drift_rho: torch.Tensor      # (B,) rho contribution (box or analytic ball)
    w_upper: torch.Tensor        # (B,) sup W over the box
    ito: torch.Tensor            # (B,) Ito trace bound
    sigma_fro: torch.Tensor      # (B,) ||Sigma||_F bound over the preimage
    jt_fro: torch.Tensor         # (B,) ||J T||_F bound over the preimage

    def certified(self) -> torch.Tensor:
        return self.upper < 0.0


# ---------------------------------------------------------------------------
# diffusion
# ---------------------------------------------------------------------------


@torch.no_grad()
def sigma_frobenius_bound(system, x_lo: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
    """Sound upper bound on ||Sigma(x)||_F for every x in the box, shape (B,)."""
    n = system.n_links
    B = x_lo.shape[0]
    if system.noise == "additive":
        return torch.full((B,), float(system.sigma) * (n ** 0.5), dtype=x_lo.dtype)
    if system.noise == "state_dependent":
        absmax = torch.maximum(x_lo[..., n:].abs(), x_hi[..., n:].abs())
        return float(system.sigma) * ((1.0 + absmax) ** 2).sum(-1).sqrt()
    if system.noise == "actuator":
        # Sigma_w = M(theta)^{-1} diag(sigma). With all angle differences inside
        # |Delta| < pi/2 we have cos >= cmin > 0, so M = L (A . cos Delta) L
        # satisfies lambda_min(M) >= cmin * lambda_min(A) * l^2 and hence
        # ||M^{-1}||_F <= n / lambda_min(M).
        theta_span = torch.maximum(x_lo[..., :n].abs(), x_hi[..., :n].abs()).max()
        cmin = max(float(torch.cos(torch.tensor(2.0 * float(theta_span)))), 1e-3)
        A = system._A()
        lam_min_A = float(torch.linalg.eigvalsh(0.5 * (A + A.T)).min())
        lam_min_M = cmin * lam_min_A * system.length ** 2
        return torch.full((B,), float(system.sigma) * n / max(lam_min_M, 1e-6), dtype=x_lo.dtype)
    raise ValueError(system.noise)


def sigma_interval(system, x_lo: torch.Tensor, x_hi: torch.Tensor) -> Iv:
    """Elementwise interval enclosure of Sigma(x) over the box, shape (B, D, N).

    Exact for the additive and state-dependent diffusions (both axis-aligned in
    velocity coordinates); not implemented for actuator noise, whose diffusion
    involves the interval inverse of the state-dependent inertia matrix.
    """
    if system.noise == "additive":
        n, s = system.n_links, system.sigma
        B = x_lo.shape[0]
        top = torch.zeros(B, n, n, dtype=x_lo.dtype)
        eye = s * torch.eye(n, dtype=x_lo.dtype).expand(B, n, n).contiguous()
        return Iv(torch.cat([top, eye], dim=-2), torch.cat([top, eye], dim=-2))
    if system.noise == "state_dependent":
        n = system.n_links
        lo_a = torch.minimum(x_lo[..., n:].abs(), x_hi[..., n:].abs())
        hi_a = torch.maximum(x_lo[..., n:].abs(), x_hi[..., n:].abs())
        s_lo = system.sigma * (1.0 + lo_a)                 # (B, n)
        s_hi = system.sigma * (1.0 + hi_a)
        top = torch.zeros(x_lo.shape[0], n, n, dtype=x_lo.dtype)
        bot_lo = torch.diag_embed(s_lo)
        bot_hi = torch.diag_embed(s_hi)
        return Iv(torch.cat([top, bot_lo], dim=-2), torch.cat([top, bot_hi], dim=-2))
    raise NotImplementedError(
        f"sigma_interval not implemented for noise={system.noise!r}; use ito_mode='cs'")


def _transpose_iv(a: Iv) -> Iv:
    return Iv(a.lo.transpose(-1, -2), a.hi.transpose(-1, -2))


@torch.no_grad()
def _ito_bound_tight(V, transport, system, kappa: float, d_eta: int, y_iv: Iv,
                     Q: torch.Tensor, hess_iv: Iv | None = None) -> tuple:
    """Tight interval Ito bound: 1/2 tr(BB^T Hess W) with B = J T(x) Sigma(x).

    The Cauchy-Schwarz bound 1/2 ||B||_F^2 ||Hess W||_F is sound but has
    irreducible slack: near the origin its slack exceeds the noise floor beta,
    so no box however small can certify against the exact threshold. The tight
    bound instead encloses BB^T itself as an interval matrix (interval Jacobian
    of the transport composed with an exact interval Sigma) and contracts it
    elementwise against the interval Hessian of W. Every elementwise product of
    intervals contains the true product, and the sum of intervals contains the
    true sum, so the result is sound -- and it converges to the exact trace as
    boxes shrink, which is what makes certification near the origin possible.

    W is block diagonal in (eta, rho) with Hess W = blkdiag(Hess V, 2 kappa Q),
    so only two contractions are needed:

        term_eta = sum_{i,j < d} BB^T[i,j] * Hess V[j,i]
        term_rho = 2 kappa sum_{i,j} BB^T[d+i,d+j] * Q[j,i]

    Returns (ito Iv[(B,)], sigma_fro, jt_fro) with the Frobenius read-outs kept
    for reporting conservatism of the removed C-S bound.
    """
    x_iv = transport.interval_inverse(y_iv)
    J_iv = transport.interval_jacobian(x_iv)               # (B, D, D)
    S_iv = sigma_interval(system, x_iv.lo, x_iv.hi)        # (B, D, N)
    Bm = _iv_matmul_batched(J_iv, S_iv)                    # (B, D, N)
    BBt = _iv_matmul_batched(Bm, _transpose_iv(Bm))        # (B, D, D)
    if hess_iv is None:
        _, _, hess_iv = V.bounds(y_iv.lo[..., :d_eta], y_iv.hi[..., :d_eta])
    # eta block: sum_ij BBt[i,j] * HessV[j,i]
    bbe = BBt[..., :d_eta, :d_eta]
    prod_e = _mul_iv(bbe, _transpose_iv(hess_iv))          # elementwise, (B, d, d)
    term_eta = Iv(prod_e.lo.sum(dim=(-1, -2)), prod_e.hi.sum(dim=(-1, -2)))
    # rho block: 2 kappa sum_ij BBt_rr[i,j] * Q[j,i] with Q exact
    bbr = BBt[..., d_eta:, d_eta:]
    Qt = Q.transpose(-1, -2).expand_as(bbr.lo)
    prod_r = _mul_iv(bbr, Iv(Qt, Qt))
    tr_rho = Iv(prod_r.lo.sum(dim=(-1, -2)), prod_r.hi.sum(dim=(-1, -2)))
    term_rho = _round(2.0 * kappa * tr_rho)
    ito = _round(0.5 * (term_eta + term_rho))
    return ito, frobenius_bound(S_iv), frobenius_bound(J_iv)


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------


@torch.no_grad()
def _res_jac(F, c: torch.Tensor, rows: slice | None = None):
    """Exact Jacobian of the (centered) residual at c, one vectorized pass.

    Returns J (B, n_out, D); rows selects a contiguous row block if given.
    jacrev differentiates a single-example function, so the batch is handled
    with vmap: J[b, o, i] = d res_o / d y_i at y = c[b].
    """
    d = F.dim
    zero = torch.zeros(d, dtype=c.dtype)
    if rows is None:
        fn = lambda y: F._residual(y, zero)
    else:
        fn = lambda y: F._residual(y, zero)[..., rows]
    with torch.enable_grad():
        J = torch.func.vmap(torch.func.jacrev(fn))(c)  # (B, n_out, D)
    return J


@torch.no_grad()
def _drift_centered(F, lo: torch.Tensor, hi: torch.Tensor, d_eta: int) -> tuple:
    """Exact drift at the box center plus a sound per-component remainder.

    Returns (F_c (B,D), rem Iv (B,D)) with F(y) in F_c + rem for every y in the
    box. The raw IBP enclosure of F is sound but its width compounds through
    the residual MLP's layers; branch-and-bound over a region-size box then
    faces slack of order hundreds and can never certify. The centered
    (mean-value) form evaluates the residual exactly at the center and bounds
    the deviation through a Jacobian enclosure whose entries are exact zeros
    (the factorised residual's eta rows do not read rho) or Lipschitz balls
    (spectral-norm products). Sound by the mean-value theorem; remainder is
    O(box radius) rather than exponentially loose.
    """
    d = F.dim
    c = 0.5 * (lo + hi)
    zero = torch.zeros(d, dtype=lo.dtype)
    res_c = F._residual(c, zero)
    with torch.enable_grad():
        J = torch.func.vmap(torch.func.jacrev(
            lambda y: F._residual(y, zero)))(c)        # (B, n_out, D)
    if F.d_eta is None:
        L = spectral_norm_bound(F.res)
        Jlo = torch.full_like(J, -L)
        Jhi = torch.full_like(J, L)
    else:
        L_eta = spectral_norm_bound(F.res_eta)
        L_rho = spectral_norm_bound(F.res_rho)
        col_e = (torch.arange(d) < d_eta).view(1, 1, d)
        row_e = (torch.arange(d) < d_eta).view(1, d, 1)
        # cascade residual: every row reads only eta -- exact zeros on the rho
        # columns, per-row Lipschitz balls on the eta columns
        Lrow = torch.where(row_e, torch.full_like(J, L_eta), torch.full_like(J, L_rho))
        Jlo = torch.where(col_e, -Lrow, torch.zeros_like(J))
        Jhi = torch.where(col_e, Lrow, torch.zeros_like(J))
    dev = Iv(lo - c, hi - c)
    rem_res = _mm_exact_batched(Jlo, Jhi, dev)
    lin_rem = iv_matmul(F.A.detach(), dev)
    rem = Iv(lin_rem.lo + rem_res.lo, lin_rem.hi + rem_res.hi)
    F_c = (c @ F.A.detach().T + res_c.detach())
    return F_c, rem


def _grad_V_centered(V, c_eta: torch.Tensor, lo_e: torch.Tensor, hi_e: torch.Tensor):
    """Exact V, grad V at the eta-box center + sound remainder widths.

    Returns (V_c, grad_c, r_lin, r_quad): sup over the box of |V - V_c - grad_c .
    (eta - c)| <= r_lin, and sup |grad V - grad_c|_i <= gfro (per-coordinate
    bound from the interval Hessian's Frobenius norm).
    """
    B, d = c_eta.shape
    ec = c_eta.clone().requires_grad_(True)
    with torch.enable_grad():
        V_c = V(ec)
        grad_c = torch.autograd.grad(V_c.sum(), ec)[0]
    _, _, hess_iv = V.bounds(lo_e, hi_e)
    gfro = torch.sqrt((hess_iv.absmax ** 2).sum(dim=(-1, -2)).clamp_min(0.0))  # (B,)
    r = 0.5 * (hi_e - lo_e)                              # (B, d) half-widths
    r_lin = (grad_c.detach().abs() * r).sum(-1) + 0.5 * gfro * (r ** 2).sum(-1)
    return V_c.detach(), grad_c.detach(), r_lin, gfro


@torch.no_grad()
def _coupling_norms(F, y_iv: Iv, d_eta: int, Q: torch.Tensor) -> torch.Tensor:
    """Sound bound on ||Q (A_re eta + g_rho(y))|| over the y-box, shape (B,).

    Centered form: g_rho(y) = res_rho(y) - res_rho(0) is enclosed as its exact
    value at the box center plus a Lipschitz-ball remainder over the box (the
    rho rows of the residual read all coordinates). The earlier version
    subtracted two interval enclosures, doubling both widths; on the coupled
    model that cancellation loss produced coupling bounds of order 1e3 and no
    certifiable box at any size.
    """
    A = F.A.detach()
    Are = A[d_eta:, :d_eta]
    eta_iv = Iv(y_iv.lo[..., :d_eta], y_iv.hi[..., :d_eta])
    lin = iv_matmul(Are, eta_iv)                         # (B, m), tight in eta
    zero = torch.zeros(F.dim, dtype=y_iv.lo.dtype)
    c = y_iv.mid
    res_c = F.residual_rho(c, zero, d_eta)             # (B, m)
    J = _res_jac(F, c, rows=slice(d_eta, None))        # (B, m, D), exact
    # cascade residual: the rho rows read only eta, so the deviation enclosure
    # has exact zeros on the rho columns and scales with the eta box only --
    # this is the property that makes b_Q shrink under eta-subdivision.
    L_rho = spectral_norm_bound(F.res_rho) if getattr(F, "d_eta", None) is not None \
        else spectral_norm_bound(F.res)
    col_e = (torch.arange(F.dim) < d_eta).view(1, 1, F.dim)
    Jlo = torch.where(col_e, torch.full_like(J, -L_rho), torch.zeros_like(J))
    Jhi = torch.where(col_e, torch.full_like(J, L_rho), torch.zeros_like(J))
    dev = Iv(y_iv.lo - c, y_iv.hi - c)                 # (B, D)
    rem = _mm_exact_batched(Jlo, Jhi, dev)             # (B, m)
    g = Iv(res_c.detach() + rem.lo, res_c.detach() + rem.hi)
    total = iv_matmul(Q, Iv(lin.lo + g.lo, lin.hi + g.hi))
    return torch.sqrt((total.absmax ** 2).sum(-1).clamp_min(0.0))


def _shell_rho_bound(b_Q: torch.Tensor, r_in: float, r_out: float):
    """max over s in [r_in, r_out] of (-s^2 + 2 s b_Q), elementwise.

    Completing the square, f(s) = b_Q^2 - (s - b_Q)^2: unimodal with maximiser
    s* = b_Q. On a shell the max is
        b_Q^2                 if r_in <= b_Q <= r_out
        f(r_in) = -r_in^2 + 2 r_in b_Q    if b_Q < r_in   (decreasing on shell)
        f(r_out) = -r_out^2 + 2 r_out b_Q if b_Q > r_out  (increasing on shell)
    """
    f_in = -r_in * r_in + 2.0 * r_in * b_Q
    f_out = -r_out * r_out + 2.0 * r_out * b_Q
    f_star = b_Q * b_Q
    return torch.maximum(torch.maximum(f_in, f_out),
                         torch.where((b_Q >= r_in) & (b_Q <= r_out), f_star, f_in))


@torch.no_grad()
def _ito_bound(V, transport, system, kappa: float, d_eta: int, y_iv: Iv,
               Q: torch.Tensor, hess_fro_eta: torch.Tensor | None = None) -> tuple:
    """Cauchy-Schwarz Ito bound (sound, loose). Kept for comparison and for the
    actuator-noise mode where the tight interval Sigma is unavailable."""
    """Sound Ito bound via Cauchy-Schwarz: 1/2 ||B||_F^2 ||Hess W||_F, B = J T Sigma."""
    m = transport.dim - d_eta
    if hess_fro_eta is None:
        hess_fro_eta = V.hessian_frobenius(y_iv.lo[..., :d_eta], y_iv.hi[..., :d_eta])
    hess_W_ub = hess_fro_eta + 2.0 * kappa * float(torch.linalg.norm(Q))
    x_iv = transport.interval_inverse(y_iv)
    jt_fro = transport.interval_jacobian_frobenius(x_iv)
    sig_fro = sigma_frobenius_bound(system, x_iv.lo, x_iv.hi)
    ito = 0.5 * (jt_fro ** 2) * (sig_fro ** 2) * hess_W_ub
    return _round(Iv(ito, ito)), sig_fro, jt_fro


# ---------------------------------------------------------------------------
# full-space certification
# ---------------------------------------------------------------------------


@torch.no_grad()
def bound_full(V, F, transport, system, kappa: float, alpha: float, d_eta: int,
               y_lo: torch.Tensor, y_hi: torch.Tensor, chunk: int = 256,
               ito_mode: str = "tight") -> BoundResult:
    """Sound bound with every coordinate searched (the unfactored baseline)."""
    Q = F.rho_metric(d_eta)
    lam_Q = float(torch.linalg.eigvalsh(Q).max())
    y_lo, y_hi = torch.atleast_2d(y_lo), torch.atleast_2d(y_hi)
    B = y_lo.shape[0]
    ups, d_etas, d_rhos, w_ups, itos, sigs, jts = [], [], [], [], [], [], []
    for s in range(0, B, chunk):
        lo, hi = y_lo[s:s + chunk], y_hi[s:s + chunk]
        y_iv = Iv(lo, hi)
        # centered drift: exact at the box center + O(radius) remainder
        F_c, F_rem = _drift_centered(F, lo, hi, d_eta)
        F_eta = Iv(F_c[..., :d_eta] + F_rem.lo[..., :d_eta],
                   F_c[..., :d_eta] + F_rem.hi[..., :d_eta])
        F_rho = Iv(F_c[..., d_eta:] + F_rem.lo[..., d_eta:],
                   F_c[..., d_eta:] + F_rem.hi[..., d_eta:])
        c_e = 0.5 * (lo[..., :d_eta] + hi[..., :d_eta])
        V_c, grad_c, _, gfro = _grad_V_centered(V, c_e, lo[..., :d_eta], hi[..., :d_eta])
        _, _, hess_iv = V.bounds(lo[..., :d_eta], hi[..., :d_eta])
        # drift_eta = grad V . F_eta, centered product of two enclosures:
        #   z = (g_c + dg).(F_c + dF) = g_c.F_c + g_c.dF + dg.F_c + dg.dF
        r_e = 0.5 * (hi[..., :d_eta] - lo[..., :d_eta])
        dg_ub = (hess_iv.absmax * r_e.unsqueeze(-2)).sum(-1)       # (B, d): |dg_i| bound
        wF_e = (F_eta.hi - F_eta.lo)
        drift_eta = ((grad_c * F_c[..., :d_eta]).sum(-1)
                     + (grad_c.abs() * wF_e).sum(-1)
                     + (dg_ub * F_c[..., :d_eta].abs()).sum(-1)
                     + (dg_ub * wF_e).sum(-1))
        rho_iv = Iv(lo[..., d_eta:], hi[..., d_eta:])
        # 2 kappa rho . Q F_rho: corner products of the *centered* F_rho
        # enclosure. The centered remainder is small, so the correlation loss of
        # the corner method is proportional to it rather than to the raw IBP
        # width -- which is what makes this term converge under subdivision.
        G_iv = iv_matmul(Q, F_rho)                     # (B, m)
        t1 = rho_iv.lo * G_iv.lo
        t2 = rho_iv.lo * G_iv.hi
        t3 = rho_iv.hi * G_iv.lo
        t4 = rho_iv.hi * G_iv.hi
        t_hi = torch.max(torch.max(t1, t2), torch.max(t3, t4))
        drift_rho = (2.0 * kappa * t_hi.sum(-1)).clamp_min(0.0)
        max_rho_sq = (torch.maximum(rho_iv.lo ** 2, rho_iv.hi ** 2)).sum(-1)
        # centered sup of V: V_c + |grad_c|.r + interval-quadratic Hessian term
        quad_hess = (hess_iv.absmax
                     * r_e.unsqueeze(-1) * r_e.unsqueeze(-2)).sum(dim=(-1, -2))
        v_up = V_c + (grad_c.abs() * r_e).sum(-1) + 0.5 * quad_hess
        w_up = v_up + kappa * lam_Q * max_rho_sq
        if ito_mode == "tight":
            ito_iv, sig_fro, jt_fro = _ito_bound_tight(V, transport, system, kappa,
                                                       d_eta, y_iv, Q, hess_iv)
        else:
            ito_iv, sig_fro, jt_fro = _ito_bound(V, transport, system, kappa, d_eta,
                                                 y_iv, Q)
        ups.append(drift_eta + drift_rho + alpha * w_up + ito_iv.hi)
        d_etas.append(drift_eta)
        d_rhos.append(drift_rho)
        w_ups.append(w_up)
        itos.append(ito_iv.hi)
        sigs.append(sig_fro)
        jts.append(jt_fro)
    cat = torch.cat
    return BoundResult(cat(ups), cat(d_etas), cat(d_rhos), cat(w_ups), cat(itos),
                       cat(sigs), cat(jts))


# ---------------------------------------------------------------------------
# factorised certification
# ---------------------------------------------------------------------------


@torch.no_grad()
def bound_factor(V, F, transport, system, kappa: float, alpha: float, d_eta: int,
                 eta_lo: torch.Tensor, eta_hi: torch.Tensor, rho_radius: float,
                 chunk: int = 256, ito_mode: str = "tight",
                 rho_rings: int = 8, r_in: float = 0.0) -> BoundResult:
    """Sound bound with only the eta-block searched and rho handled analytically.

    The rho ball is partitioned into `rho_rings` shells [r_in_k, r_out_k]; the
    caller (the bound factory) evaluates each shell and takes the worst, which
    is sound because the ball is the union of the shells. Passing rho_rings=1
    with r_in=0 recovers the single full-ball bound. Shell bounds matter: on the
    full ball the completing-the-square term kappa*b_Q^2 and the alpha*W ball
    term are independent of the eta box and dominate everything, so no box could
    certify; on inner shells both shrink with the shell radius.
    """
    Q = F.rho_metric(d_eta)
    # W's transversal term is a Lyapunov certificate only if Q is positive
    # definite. The metric machinery falls back to the identity (which is PD)
    # when A_rr is not Hurwitz, so Q here is always PD; what the fallback loses
    # is the exact identity A^T Q + Q A = -I that the completing-the-square
    # step of the rho bound uses. Certifying with a fallback metric would
    # therefore be unsound, so this bound refuses: +inf means "no certificate
    # available for this model", which is a result and not a crash.
    A_rr = F.A.detach()[d_eta:, d_eta:]
    resid = (A_rr.T @ Q + Q @ A_rr + torch.eye(Q.shape[0], dtype=Q.dtype)).abs().max()
    if float(resid) > 1e-6:
        inf = torch.full((4,), float("inf"), dtype=Q.dtype)
        return BoundResult(inf, inf.clone(), inf.clone(), inf.clone(), inf.clone(),
                           inf.clone(), inf.clone())
    lam_Q = float(torch.linalg.eigvalsh(Q).max())
    m = transport.dim - d_eta
    r = float(rho_radius)
    eta_lo, eta_hi = torch.atleast_2d(eta_lo), torch.atleast_2d(eta_hi)
    B = eta_lo.shape[0]
    ups, d_etas, d_rhos, w_ups, itos, sigs, jts = [], [], [], [], [], [], []
    for s in range(0, B, chunk):
        lo_e, hi_e = eta_lo[s:s + chunk], eta_hi[s:s + chunk]
        lo = torch.cat([lo_e, torch.full((lo_e.shape[0], m), -r, dtype=lo_e.dtype)], dim=-1)
        hi = torch.cat([hi_e, torch.full((hi_e.shape[0], m), r, dtype=hi_e.dtype)], dim=-1)
        y_iv = Iv(lo, hi)
        # centered drift over the full (eta box x ball-as-box) enclosure
        F_c, F_rem = _drift_centered(F, lo, hi, d_eta)
        F_eta = Iv(F_c[..., :d_eta] + F_rem.lo[..., :d_eta],
                   F_c[..., :d_eta] + F_rem.hi[..., :d_eta])
        V_c, grad_c, _, gfro = _grad_V_centered(V, 0.5 * (lo_e + hi_e), lo_e, hi_e)
        _, _, hess_iv = V.bounds(lo_e, hi_e)
        r_e = 0.5 * (hi_e - lo_e)
        dg_ub = (hess_iv.absmax * r_e.unsqueeze(-2)).sum(-1)
        wF_e = F_eta.hi - F_eta.lo
        drift_eta = ((grad_c * F_c[..., :d_eta]).sum(-1)
                     + (grad_c.abs() * wF_e).sum(-1)
                     + (dg_ub * F_c[..., :d_eta].abs()).sum(-1)
                     + (dg_ub * wF_e).sum(-1))
        b_Q = _coupling_norms(F, y_iv, d_eta, Q)
        rho_free = _shell_rho_bound(b_Q, r_in, r)
        drift_rho = kappa * rho_free + alpha * kappa * lam_Q * r * r
        quad_hess = (hess_iv.absmax
                     * r_e.unsqueeze(-1) * r_e.unsqueeze(-2)).sum(dim=(-1, -2))
        v_up = V_c + (grad_c.abs() * r_e).sum(-1) + 0.5 * quad_hess
        w_up = v_up + kappa * lam_Q * r * r
        if ito_mode == "tight":
            ito_iv, sig_fro, jt_fro = _ito_bound_tight(V, transport, system, kappa,
                                                       d_eta, y_iv, Q, hess_iv)
        else:
            ito_iv, sig_fro, jt_fro = _ito_bound(V, transport, system, kappa, d_eta,
                                                 y_iv, Q)
        ups.append(drift_eta + drift_rho + alpha * v_up + ito_iv.hi)
        d_etas.append(drift_eta)
        d_rhos.append(drift_rho)
        w_ups.append(w_up)
        itos.append(ito_iv.hi)
        sigs.append(sig_fro)
        jts.append(jt_fro)
    cat = torch.cat
    return BoundResult(cat(ups), cat(d_etas), cat(d_rhos), cat(w_ups), cat(itos),
                       cat(sigs), cat(jts))


@torch.no_grad()
def ito_tightness(V, F, transport, system, kappa: float, d_eta: int,
                  y: torch.Tensor, ito_mode: str = "tight") -> dict:
    """Compare the sound Ito bound against the exact trace at sampled points."""
    Q = F.rho_metric(d_eta)
    yv = y.clone().requires_grad_(True)
    W = V(yv[..., :d_eta]) + kappa * ((yv[..., d_eta:] @ Q) * yv[..., d_eta:]).sum(-1)
    gW = torch.autograd.grad(W.sum(), yv, create_graph=True)[0]
    x = transport.inverse(yv)
    xr = x.clone().requires_grad_(True)
    J = torch.stack([torch.autograd.grad(transport(xr)[..., k].sum(), xr,
                                         create_graph=True, retain_graph=True)[0]
                     for k in range(transport.dim)], dim=-2)
    sig = system.diffusion(xr)
    Bm = J @ sig
    BBt = Bm @ Bm.transpose(-1, -2)
    D = y.shape[-1]
    ito_exact = torch.zeros(y.shape[0], dtype=y.dtype)
    for i in range(D):
        gi = torch.autograd.grad(gW[:, i].sum(), yv, create_graph=True, retain_graph=True)[0]
        ito_exact = ito_exact + 0.5 * (gi * BBt[:, i, :]).sum(-1)
    y_iv = Iv(y, y)
    bound = _ito_bound_tight if ito_mode == "tight" else _ito_bound
    ito_iv, _, _ = bound(V, transport, system, kappa, d_eta, y_iv, Q)
    ratio = ito_iv.hi / ito_exact.detach().abs().clamp_min(1e-12)
    return {"bound_median": float(ito_iv.hi.median()),
            "exact_median": float(ito_exact.detach().median()),
            "median_ratio": float(ratio.median()),
            "max_ratio": float(ratio.max())}
