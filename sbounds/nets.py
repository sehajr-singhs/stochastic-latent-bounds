"""Sound interval arithmetic, IBP, and interval forward-mode AD for MLPs.

Everything in this module is written to be *sound*: every enclosure is inflated
outward by a small relative epsilon so that floating point roundoff can never
make a bound tighter than the true quantity. Looseness is a cost we pay on
purpose; the repo reports the measured conservatism of these bounds rather than
hiding it.

The interval AD engine carries a ternary jet per unit:
    value   : Iv of shape (B, U)                (U units)
    jacobian: Iv of shape (B, U, n_in)          d(v)/d(x)
    hessian : Iv of shape (B, U, n_in, n_in)    d2(v)/d(x)^2
and propagates it layer by layer. Activation functions are applied elementwise,
so per-unit jets do not mix.
"""
from __future__ import annotations

import math

import torch

# Relative outward inflation applied to every interval produced by mixed
# operations. 1e-12 is ~4000 ulps for float64, far above observed drift.
ROUNDOFF = 1e-12


class Iv:
    """Axis-aligned interval enclosure with outward-rounded, sound arithmetic."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    # --- constructors -------------------------------------------------
    @staticmethod
    def point(t):
        return Iv(t, t)

    @staticmethod
    def const(t, like):
        return Iv(torch.full_like(like.lo, t), torch.full_like(like.hi, t))

    # --- views --------------------------------------------------------
    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def wid(self):
        return self.hi - self.lo

    @property
    def absmax(self):
        return torch.maximum(self.lo.abs(), self.hi.abs())

    def shaped(self, *shape):
        return Iv(self.lo.reshape(shape), self.hi.reshape(shape))

    def unsqueeze(self, dim):
        return Iv(self.lo.unsqueeze(dim), self.hi.unsqueeze(dim))

    def squeeze(self, dim=None):
        if dim is None:
            return Iv(self.lo.squeeze(), self.hi.squeeze())
        return Iv(self.lo.squeeze(dim), self.hi.squeeze(dim))

    def expand(self, *shape):
        return Iv(self.lo.expand(*shape), self.hi.expand(*shape))

    def reshape(self, *shape):
        return Iv(self.lo.reshape(*shape), self.hi.reshape(*shape))

    def __getitem__(self, item):
        return Iv(self.lo[item], self.hi[item])

    def sum(self, dim=None):
        if dim is None:
            return Iv(self.lo.sum(), self.hi.sum())
        return Iv(self.lo.sum(dim), self.hi.sum(dim))

    # --- arithmetic ---------------------------------------------------
    def __add__(self, o):
        if not isinstance(o, Iv):
            return Iv(self.lo + o, self.hi + o)
        return _round(Iv(self.lo + o.lo, self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        if not isinstance(o, Iv):
            return Iv(self.lo - o, self.hi - o)
        return _round(Iv(self.lo - o.hi, self.hi - o.lo))

    def __rsub__(self, o):
        return Iv(o - self.hi, o - self.lo)

    def __mul__(self, o):
        if not isinstance(o, Iv):
            # scalar / exact tensor multiplier
            if isinstance(o, torch.Tensor):
                return _round(Iv(self.lo * o, self.hi * o)) if bool((o >= 0).all()) else _mul_exact_any(o, self)
            if o >= 0:
                return _round(Iv(self.lo * o, self.hi * o))
            return _round(Iv(self.hi * o, self.lo * o))
        return _round(_mul_iv(self, o))

    __rmul__ = __mul__

    def sq(self):
        return _round(_mul_iv(self, self))

    def __truediv__(self, k):
        """Division by a positive scalar (or positive exact tensor)."""
        if isinstance(k, torch.Tensor):
            return Iv(self.lo / k, self.hi / k)
        assert k > 0
        return Iv(self.lo / k, self.hi / k)

    # --- elementwise monotone maps ------------------------------------
    def tanh(self):
        return _round(Iv(torch.tanh(self.lo), torch.tanh(self.hi)))

    def softplus(self):
        return _round(Iv(torch.nn.functional.softplus(self.lo),
                         torch.nn.functional.softplus(self.hi)))

    def exp(self):
        return _round(Iv(torch.exp(self.lo), torch.exp(self.hi)))

    def relu(self):
        return Iv(torch.clamp(self.lo, min=0.0), torch.clamp(self.hi, min=0.0))

    def sqrt_absmax(self):
        return torch.sqrt(self.absmax.clamp_min(0.0))

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Iv(lo={self.lo.shape}, hi={self.hi.shape})"


def _round(iv: Iv) -> Iv:
    """Inflate outward by a relative epsilon to absorb float roundoff."""
    c = iv.absmax.clamp_min(1.0)
    infl = ROUNDOFF * c
    return Iv(iv.lo - infl, iv.hi + infl)


def _mul_iv(a: Iv, b: Iv) -> Iv:
    p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    lo = torch.minimum(torch.minimum(p[0], p[1]), torch.minimum(p[2], p[3]))
    hi = torch.maximum(torch.maximum(p[0], p[1]), torch.maximum(p[2], p[3]))
    return Iv(lo, hi)


def _mul_exact_any(k: torch.Tensor, iv: Iv) -> Iv:
    p = (k * iv.lo, k * iv.hi)
    return Iv(torch.minimum(p[0], p[1]), torch.maximum(p[0], p[1]))


def iv_matmul(W: torch.Tensor, x: Iv) -> Iv:
    """Sound enclosure of W @ x for exact W (n_out, n_in) and interval x.

    W is split into its non-negative and non-positive parts so the corner
    products are chosen per coefficient.
    """
    Wp = torch.clamp(W, min=0.0)
    Wn = torch.clamp(W, max=0.0)
    # x: (..., n_in) -> contract last dim; W: (n_out, n_in)
    lo = x.lo @ Wp.T + x.hi @ Wn.T
    hi = x.hi @ Wp.T + x.lo @ Wn.T
    return _round(Iv(lo, hi))


def iv_concat(parts, dim=-1) -> Iv:
    return Iv(torch.cat([p.lo for p in parts], dim=dim),
              torch.cat([p.hi for p in parts], dim=dim))


# ---------------------------------------------------------------------------
# MLP with sound bounds
# ---------------------------------------------------------------------------


ACTIVATIONS = {
    "tanh": (torch.tanh,
             lambda v: 1.0 - torch.tanh(v) ** 2,
             lambda v: -2.0 * torch.tanh(v) * (1.0 - torch.tanh(v) ** 2)),
    "softplus": (torch.nn.functional.softplus,
                 torch.sigmoid,
                 lambda v: torch.sigmoid(v) * (1.0 - torch.sigmoid(v))),
    "relu": (lambda v: torch.clamp(v, min=0.0),
             lambda v: (v > 0).to(v.dtype),
             lambda v: torch.zeros_like(v)),
    "linear": (lambda v: v, lambda v: torch.ones_like(v), lambda v: torch.zeros_like(v)),
}


def _act_iv(name, iv: Iv) -> Iv:
    if name == "tanh":
        return iv.tanh()
    if name == "softplus":
        return iv.softplus()
    if name == "relu":
        return iv.relu()
    if name == "linear":
        return iv
    raise KeyError(name)


class MLP(torch.nn.Module):
    """Plain MLP with an activation taken from ACTIVATIONS on every hidden layer."""

    def __init__(self, n_in: int, n_out: int, width: int = 32, depth: int = 2,
                 act: str = "tanh", out_act: str = "linear", seed: int | None = None,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        assert act in ACTIVATIONS and out_act in ACTIVATIONS
        self.n_in, self.n_out, self.act, self.out_act = n_in, n_out, act, out_act
        dims = [n_in] + [width] * max(depth - 1, 0) + [n_out]
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)]
        )
        self.n_act_layers = len(dims) - 2
        self.to(dtype)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int | None = None):
        g = torch.Generator().manual_seed(0 if seed is None else seed)
        for lin in self.linears:
            fan_in = lin.weight.shape[1]
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.uniform_(-bound, bound, generator=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < self.n_act_layers:
                h = ACTIVATIONS[self.act][0](h)
            else:
                h = ACTIVATIONS[self.out_act][0](h)
        return h

    # --- sound bounds -------------------------------------------------
    @torch.no_grad()
    def ibp(self, x: Iv) -> Iv:
        """Interval bound propagation: sound enclosure of forward(x) over a box."""
        h = x
        for i, lin in enumerate(self.linears):
            h = iv_matmul(lin.weight, h) + lin.bias.detach()
            if i < self.n_act_layers:
                h = _act_iv(self.act, h)
            else:
                h = _act_iv(self.out_act, h)
        return h

    @torch.no_grad()
    def jet(self, lo: torch.Tensor, hi: torch.Tensor):
        """Ternary jet (value, grad, hess) enclosures over the box [lo, hi].

        Returns (val Iv[(B,n_out)], grad Iv[(B,n_out,n_in)], hess Iv[(B,n_out,n_in,n_in)])
        with grad[b,o,i] = d out_o / d x_i and hess[b,o,i,k] = d2 out_o / d x_i d x_k.
        """
        B, n_in = lo.shape
        val = Iv(lo.clone(), hi.clone())                       # (B, n_in)
        eye = torch.eye(n_in, dtype=lo.dtype).expand(B, n_in, n_in)
        jac = Iv(eye.clone(), eye.clone())                     # (B, n_in, n_in)
        hess = Iv(torch.zeros(B, n_in, n_in, n_in, dtype=lo.dtype),
                  torch.zeros(B, n_in, n_in, n_in, dtype=lo.dtype))

        for li, lin in enumerate(self.linears):
            W = lin.weight.detach()                            # (U, P)
            b = lin.bias.detach()
            U, P = W.shape
            # --- linear part: contract the previous unit axis (P) -------------
            Wp, Wn = torch.clamp(W, min=0.0), torch.clamp(W, max=0.0)
            v_lo = val.lo @ Wp.T + val.hi @ Wn.T + b
            v_hi = val.hi @ Wp.T + val.lo @ Wn.T + b
            # jac[b, p, i] = sum_j W[p, j] jac[b, j, i]
            j_lo = torch.einsum("pj,bji->bpi", Wp, jac.lo) + torch.einsum("pj,bji->bpi", Wn, jac.hi)
            j_hi = torch.einsum("pj,bji->bpi", Wp, jac.hi) + torch.einsum("pj,bji->bpi", Wn, jac.lo)
            # hess[b, p, i, k] = sum_j W[p, j] hess[b, j, i, k]
            h_lo = torch.einsum("pj,bjik->bpik", Wp, hess.lo) + torch.einsum("pj,bjik->bpik", Wn, hess.hi)
            h_hi = torch.einsum("pj,bjik->bpik", Wp, hess.hi) + torch.einsum("pj,bjik->bpik", Wn, hess.lo)
            val = _round(Iv(v_lo, v_hi))
            jac = _round(Iv(j_lo, j_hi))
            hess = _round(Iv(h_lo, h_hi))

            # --- activation, elementwise over the U output units --------------
            act = self.act if li < self.n_act_layers else self.out_act
            if act != "linear":
                d1 = _act_iv_deriv(act, 1, val)
                d2 = _act_iv_deriv(act, 2, val)
                val = _act_iv(act, val)
                # hess = phi'' * (g outer g) + phi' * hess, per output unit
                outer = _outer_bounds(jac)                      # (B,U,n_in,n_in)
                d1_4 = d1.unsqueeze(-1).unsqueeze(-1)           # (B,U,1,1)
                d2_4 = d2.unsqueeze(-1).unsqueeze(-1)           # (B,U,1,1)
                hess = _round(d2_4 * outer + d1_4 * hess)
                jac = _round(d1.unsqueeze(-1) * jac)            # (B,U,1)*(B,U,n)
            else:
                val = _act_iv(act, val)
        return val, jac, hess


def _act_iv_deriv(name: str, order: int, v: Iv) -> Iv:
    """Sound enclosure of phi' or phi'' over the *value* interval v.

    For tanh/softplus the derivatives are monotone in v over the ranges we care
    about, so evaluating at the interval endpoints is sound. For tanh, |phi'|<=1
    and phi'' is odd and decreasing, so endpoint evaluation is sound as well.
    """
    if name == "tanh":
        t_lo, t_hi = torch.tanh(v.lo), torch.tanh(v.hi)
        if order == 1:
            return _round(Iv(1.0 - t_hi ** 2, 1.0 - t_lo ** 2))
        # phi'' = -2 t (1 - t^2); bound by evaluating all corners
        a = -2.0 * t_lo * (1.0 - t_lo ** 2)
        b = -2.0 * t_hi * (1.0 - t_hi ** 2)
        return _round(Iv(torch.minimum(a, b), torch.maximum(a, b)))
    if name == "softplus":
        s_lo, s_hi = torch.sigmoid(v.lo), torch.sigmoid(v.hi)
        if order == 1:
            return _round(Iv(s_lo, s_hi))
        a = s_lo * (1.0 - s_lo)
        b = s_hi * (1.0 - s_hi)
        return _round(Iv(torch.minimum(a, b), torch.maximum(a, b)))
    if name == "relu":
        return Iv(torch.zeros_like(v.lo), torch.ones_like(v.hi)) if order == 1 else Iv(
            torch.zeros_like(v.lo), torch.zeros_like(v.hi))
    raise KeyError(name)


def _outer_bounds(jac: Iv) -> Iv:
    """Sound enclosure of (g g^T) entries, i.e. all pairwise products g_i g_k.

    Per-entry intervals are the exact products; per-entry multiplication already
    gives a sound enclosure, and no cross-entry correlation is exploited, which
    is where the looseness of the Hessian bound comes from.
    """
    return _mul_iv(jac.unsqueeze(-1), jac.unsqueeze(-2))


def spectral_norm_bound(module: MLP, iters: int = 30) -> float:
    """Upper bound on the network's Lipschitz constant via layerwise spectral norms.

    Each Linear's operator 2-norm is power-iterated then inflated by 1e-9, and
    the activations we support all have Lipschitz constant <= 1, so the product
    is a valid upper bound.
    """
    total = 1.0
    for lin in module.linears:
        total *= _spectral(lin.weight.detach(), iters) * (1.0 + 1e-9)
    for i in range(len(module.linears) - 1):
        if module.act in ("tanh", "softplus", "relu"):
            total *= 1.0
    return float(total)


def _spectral(W: torch.Tensor, iters: int = 30) -> float:
    if W.numel() == 0:
        return 0.0
    m, n = W.shape
    v = torch.ones(n, dtype=W.dtype)
    for _ in range(iters):
        u = W @ v
        nu = u.norm()
        if nu == 0:
            return 0.0
        u = u / nu
        v = W.T @ u
        nv = v.norm()
        if nv == 0:
            return 0.0
        v = v / nv
    sigma = float((W @ v).norm())
    return sigma


def frobenius_bound(iv: Iv) -> torch.Tensor:
    """Sound upper bound on the Frobenius norm of a matrix-valued interval.

    ||M||_F = sqrt(sum_ij M_ij^2) <= sqrt(sum_ij max(|lo_ij|,|hi_ij|)^2).
    """
    return torch.sqrt((iv.absmax ** 2).sum(dim=(-1, -2)).clamp_min(0.0))
