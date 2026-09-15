"""Invertible transport: a learned diffeomorphism of R^D that factors the state
into a low-dimensional certified factor eta and a transversal residual rho.

A diffeomorphism of R^D cannot map R^D onto R^d for d < D, so the honest claim is
not "compress dimensions away" but "factor them". The transport T: R^D -> R^D is
exactly invertible with T(x*) = 0, and the certificate is built on the eta-block
only, treating rho as a bounded transversal residual whose contraction is proved
analytically rather than by search. The dimension reduction is real, but it comes
from the *decomposition*, not from a fictitious low-dimensional bijection. The
repo states this plainly rather than claiming an invertible R^D -> R^d encoder.

Implementation: affine coupling layers (RealNVP) with tanh-bounded log-scales, so
exp(s) is always in a known compact interval and the Jacobian is never degenerate.
The interval machinery below is what makes the certificate sound: it propagates
interval enclosures of T, T^{-1} and ||J T||_F so the Ito trace in physical
coordinates can be bounded while the search happens in latent coordinates.
"""
from __future__ import annotations

import torch

from .nets import Iv, MLP, _round, iv_matmul


class CouplingLayer(torch.nn.Module):
    """Affine coupling with a tanh-bounded log-scale.

    mask[i] == 1 marks coordinates carried through unchanged; mask[i] == 0 marks
    coordinates transformed as x_i -> x_i * exp(s_i) + t_i.
    """

    def __init__(self, dim: int, mask: torch.Tensor, width: int = 32, hidden_depth: int = 2,
                 act: str = "tanh", scale_max: float = 1.0, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.to(torch.float64))
        self.scale_max = float(scale_max)
        self.s_net = MLP(dim, dim, width, hidden_depth + 1, act, "linear", seed=seed)
        self.t_net = MLP(dim, dim, width, hidden_depth + 1, act, "linear", seed=seed + 977)
        # start near identity: zero the scale branch's last layer
        with torch.no_grad():
            self.s_net.linears[-1].weight.mul_(0.0)
            self.s_net.linears[-1].bias.mul_(0.0)
            self.t_net.linears[-1].weight.mul_(0.0)
            self.t_net.linears[-1].bias.mul_(0.0)

    # --- exact maps ---------------------------------------------------
    def _st(self, a: torch.Tensor):
        s = self.scale_max * torch.tanh(self.s_net(a))
        t = self.t_net(a)
        return s, t

    def forward(self, x: torch.Tensor):
        a = x * self.mask
        s, t = self._st(a)
        nm = 1.0 - self.mask
        y = a + nm * (x * torch.exp(s) + t)
        return y

    def inverse(self, y: torch.Tensor):
        a = y * self.mask
        s, t = self._st(a)
        nm = 1.0 - self.mask
        x = a + nm * ((y - t) * torch.exp(-s))
        return x

    def logdet(self, x: torch.Tensor):
        a = x * self.mask
        s, _ = self._st(a)
        return (s * (1.0 - self.mask)).sum(-1)

    # --- sound interval maps -----------------------------------------
    def interval_value(self, x: Iv) -> Iv:
        m, nm = self.mask, 1.0 - self.mask
        a = x * m
        s_raw = self.s_net.jet(a.lo, a.hi)[0]
        s = (s_raw * self.scale_max).tanh() * self.scale_max
        t_val = self.t_net.jet(a.lo, a.hi)[0]
        return _round(a + (x * s.exp() + t_val) * nm)

    def interval_inverse(self, y: Iv) -> Iv:
        m, nm = self.mask, 1.0 - self.mask
        a = y * m
        s_raw = self.s_net.jet(a.lo, a.hi)[0]
        s = (s_raw * self.scale_max).tanh() * self.scale_max
        t_val = self.t_net.jet(a.lo, a.hi)[0]
        return _round(a + ((y - t_val) * (-s).exp()) * nm)

    def interval_jacobian(self, x: Iv):
        """Sound enclosure of J = d y / d x as an Iv of shape (B, D, D)."""
        m, nm = self.mask, 1.0 - self.mask
        a = x * m
        jet_s = self.s_net.jet(a.lo, a.hi)
        s_raw_val, s_raw_jac = jet_s[0], jet_s[1]
        jet_t = self.t_net.jet(a.lo, a.hi)
        t_val, t_jac = jet_t[0], jet_t[1]
        th = (s_raw_val * self.scale_max).tanh()
        s = th * self.scale_max
        # d s_i / d a_j = scale_max * (1 - tanh^2)_i * (d s_raw_i / d a_j),
        # shaped (B, i, j): the per-row factor goes on the row axis explicitly
        dsda = ((1.0 - th.sq()) * self.scale_max).unsqueeze(-1) * s_raw_jac
        E = s.exp()                                            # (B, D)
        # Rows for i in mask are delta_ij. Rows for i off-mask are
        #   d y_i / d x_j = delta_ij exp(s_i) + m_j ( x_i exp(s_i) ds_i/da_j + dt_i/da_j )
        B, D = x.lo.shape
        m_b = m.unsqueeze(0).expand(B, D)
        nm_b = nm.unsqueeze(0).expand(B, D)
        diag = _round(Iv(m_b + nm_b * E.lo, m_b + nm_b * E.hi))
        J_lo = torch.diag_embed(diag.lo)
        J_hi = torch.diag_embed(diag.hi)
        term = x.unsqueeze(-1) * E.unsqueeze(-1) * dsda + t_jac      # (B, D, D)
        term = term * m.unsqueeze(0).unsqueeze(0)                    # factor m_j
        term = term * nm.unsqueeze(0).unsqueeze(-1)                  # rows i off-mask only
        return _round(Iv(J_lo + term.lo, J_hi + term.hi))


class InvertibleTransport(torch.nn.Module):
    """T(x) = R((x - x*) / x_scale) - R(0), a diffeomorphism with T(x*) = 0."""

    def __init__(self, dim: int, d_latent: int, n_layers: int = 4, width: int = 32,
                 hidden_depth: int = 2, act: str = "tanh", scale_max: float = 1.0,
                 x_star: torch.Tensor | None = None, x_scale: torch.Tensor | None = None,
                 seed: int = 0):
        super().__init__()
        self.dim = dim
        self.d_latent = d_latent
        self.register_buffer("x_star", torch.zeros(dim, dtype=torch.float64)
                             if x_star is None else x_star.to(torch.float64))
        self.register_buffer("x_scale", torch.ones(dim, dtype=torch.float64)
                             if x_scale is None else x_scale.to(torch.float64))
        half = (dim + 1) // 2
        self.layers = torch.nn.ModuleList()
        for l in range(n_layers):
            mask = torch.zeros(dim, dtype=torch.float64)
            if l % 2 == 0:
                mask[:half] = 1.0
            else:
                mask[half:] = 1.0
            self.layers.append(CouplingLayer(dim, mask, width, hidden_depth, act,
                                             scale_max, seed=seed + 31 * l))
        self._r0 = None

    # --- exact maps ---------------------------------------------------
    def _norm(self, x):
        return (x - self.x_star) / self.x_scale

    def _denorm(self, xn):
        return xn * self.x_scale + self.x_star

    def r_zero(self) -> torch.Tensor:
        z = torch.zeros(self.dim, dtype=torch.float64, device=self.x_star.device)
        h = z
        for lay in self.layers:
            h = lay(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._norm(x)
        for lay in self.layers:
            h = lay(h)
        return h - self.r_zero().to(h.dtype)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        h = y + self.r_zero().to(y.dtype)
        for lay in reversed(self.layers):
            h = lay.inverse(h)
        return self._denorm(h)

    def logdet(self, x: torch.Tensor) -> torch.Tensor:
        h = self._norm(x)
        acc = torch.zeros(h.shape[:-1], dtype=h.dtype)
        for lay in self.layers:
            acc = acc + lay.logdet(h)
            h = lay(h)
        return acc - self.x_scale.log().sum()

    # --- sound interval maps -----------------------------------------
    @torch.no_grad()
    def interval_forward(self, x: Iv) -> Iv:
        h = self._norm(x)
        for lay in self.layers:
            h = lay.interval_value(h)
        r0 = self.r_zero()
        return _round(h - r0)

    @torch.no_grad()
    def interval_inverse(self, y: Iv) -> Iv:
        r0 = self.r_zero()
        h = y + r0
        for lay in reversed(self.layers):
            h = lay.interval_inverse(h)
        return _round(self._denorm(h))

    @torch.no_grad()
    def interval_jacobian_frobenius(self, x: Iv) -> torch.Tensor:
        """Sound upper bound on ||J T(x)||_F for every x in the interval x.

        The chain rule for a composition needs a matrix product of interval
        Jacobians; we follow the same convention as the value propagation and
        keep the enclosure sound, if loose.
        """
        # Start with J = diag(1 / x_scale) since x_n = (x - x*)/x_scale
        B, D = x.lo.shape
        J = Iv(torch.diag_embed((1.0 / self.x_scale).expand(B, D).contiguous()),
               torch.diag_embed((1.0 / self.x_scale).expand(B, D).contiguous()))
        h = self._norm(x)
        for lay in self.layers:
            Jlay = lay.interval_jacobian(h)
            J = _iv_matmul_batched(Jlay, J)
            h = lay.interval_value(h)
        return torch.sqrt((J.absmax ** 2).sum(dim=(-1, -2)).clamp_min(0.0))

    def interval_jacobian(self, x: Iv) -> Iv:
        B, D = x.lo.shape
        J = Iv(torch.diag_embed((1.0 / self.x_scale).expand(B, D).contiguous()),
               torch.diag_embed((1.0 / self.x_scale).expand(B, D).contiguous()))
        h = self._norm(x)
        for lay in self.layers:
            J = _iv_matmul_batched(lay.interval_jacobian(h), J)
            h = lay.interval_value(h)
        return J

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Exact Jacobian via autograd, shape (B, D, D)."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            outs = []
            for k in range(self.dim):
                g = torch.autograd.grad(self.forward(x)[..., k].sum(), x, retain_graph=True)[0]
                outs.append(g)
            return torch.stack(outs, dim=-2)

    def lipschitz_estimate(self, scale: float = 1.0) -> float:
        ico = Iv(self.x_star - self.x_scale * scale, self.x_star + self.x_scale * scale)
        return float(self.interval_jacobian_frobenius(ico).max())


def _iv_matmul_batched(A: Iv, Bm: Iv) -> Iv:
    """Sound enclosure of the batched matrix product A @ B, shapes (..., n, k), (..., k, m)."""
    A3, B3 = A.lo.unsqueeze(-1), Bm.lo.unsqueeze(-3)
    A4, B4 = A.hi.unsqueeze(-1), Bm.hi.unsqueeze(-3)
    p = (A3 * B3, A3 * B4, A4 * B3, A4 * B4)
    lo = torch.minimum(torch.minimum(p[0], p[1]), torch.minimum(p[2], p[3])).sum(-2)
    hi = torch.maximum(torch.maximum(p[0], p[1]), torch.maximum(p[2], p[3])).sum(-2)
    return _round(Iv(lo, hi))
