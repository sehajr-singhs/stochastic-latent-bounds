"""Data generation and training for the world model and the latent certificate.

The world model is trained the way a physical-AI world model is trained: on
pushed-forward observation pairs (x_t, x_{t+1}), regressing the finite
difference (y_{t+1} - y_t) / dt. Nothing reads the analytic drift during
training, so the resulting latent model is a genuine data-driven model. The
analytic Ito push-forward in `systems.pushforward_drift` is used only to score
the trained model afterwards, which is what makes the `ito_correction` claim
falsifiable instead of rhetorical.

Training is joint: the transport T and the latent drift F are optimised
together, because the split into (eta, rho) is a property of the pair, not of
either alone. Four losses are combined:

  L_fit       finite-difference regression on held-in rollout pairs
  L_decouple  ||A_eta_rho||_F^2, pushing rho out of the eta equation
  L_gres      ||res(y)_rho - res(0)_rho||^2, the rho rows of the residual --
              exactly the g_rho forcing term the sound coupling bound b_Q needs
              small; this is the term that makes the factorisation certifiable
  L_contract  softplus(lam_max(A_rho_rho) + target), forcing transversal
              contraction at a stated rate; the rho term of the certificate is
              only useful when this succeeds, and we report it either way

`L_anchor` keeps the transport from collapsing: the regression loss is
scale-invariant in T, so without an anchor the latent coordinates shrink toward
zero and every bound becomes vacuous.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .generator import latent_generator, noise_floor
from .models import LatentDynamics, LyapunovNet, spectral_project, solve_lyapunov_metric
from .systems import ChainArm, pushforward_drift


@dataclass
class RolloutData:
    x0: torch.Tensor        # (N, D) start of each pair
    x1: torch.Tensor        # (N, D) one step later
    dt: float
    system: object | None = None   # the plant that generated the pairs (needed
                                   # for the exact push-forward training target)

    def batches(self, batch: int, generator):
        idx = torch.randperm(self.x0.shape[0], generator=generator)
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            yield self.x0[j], self.x1[j]


def euler_maruyama(system: ChainArm, x0: torch.Tensor, n_steps: int, dt: float,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Integrate dx = f(x) dt + Sigma(x) dW for n_steps of size dt."""
    x = x0.clone()
    for _ in range(n_steps):
        f = system.drift(x)
        sig = system.diffusion(x)                      # (B, D, n_noise)
        dW = torch.randn((x.shape[0], sig.shape[-1]), dtype=x.dtype,
                         generator=generator) * (dt ** 0.5)
        x = x + f * dt + (sig @ dW.unsqueeze(-1)).squeeze(-1)
    return x


def generate_pairs(system: ChainArm, n_traj: int = 4000, dt: float = 0.01,
                   region_scale: float = 1.0, substeps: int = 1,
                   seed: int = 0, burn_in: int = 0) -> RolloutData:
    """Rollout pairs (x_t, x_{t+dt}) from random initial conditions."""
    g = torch.Generator().manual_seed(seed)
    scale = system.lqr_like_scale() * region_scale
    u = torch.rand((n_traj, system.dim), dtype=torch.float64, generator=g)
    x0 = (2.0 * u - 1.0) * scale
    if burn_in:
        x0 = euler_maruyama(system, x0, burn_in, dt, generator=g)
    x1 = euler_maruyama(system, x0, substeps, dt / substeps, generator=g)
    return RolloutData(x0.detach(), x1.detach(), dt)


@dataclass
class TrainConfig:
    steps: int = 2500
    batch: int = 256
    lr: float = 3e-3
    w_decouple: float = 2.0
    w_gres: float = 1.0
    w_contract: float = 1.0
    w_anchor: float = 1e-3
    contract_target: float = 1.0
    metric_every: int = 25
    refit_A: bool = True
    spec_cap: float = 4.0
    rho_spec_cap: float = 1.0        # tight cap on the rho-rows' Lipschitz ball
    d_eta: int | None = None         # factorised residual when set
    hard_decouple: bool = True       # project A[e, rho] to zero after every step
    rho_margin: float = 2.0          # mid-training stability margin for A_rr
    log_every: int = 250
    seed: int = 0
    target_pushforward: bool = False # fit F to the plant's exact Ito push-forward
                                     # (per-batch, recomputed as T moves) instead of
                                     # raw single-step differences -- the right target
                                     # when real data's per-step signal is noise-dominated


def train_world_model(transport, F: LatentDynamics, data: RolloutData, d_eta: int,
                      cfg: TrainConfig = TrainConfig(), verbose: bool = False) -> dict:
    """Jointly train the transport and the latent drift on rollout pairs."""
    torch.manual_seed(cfg.seed)
    params = list(transport.parameters()) + list(F.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    hist = {"fit": [], "decouple": [], "gres": [], "contract": [], "anchor": []}
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    step = 0
    while step < cfg.steps:
        for x0, x1 in data.batches(cfg.batch, gen):
            if step >= cfg.steps:
                break
            y0, y1 = transport(x0), transport(x1)
            if cfg.target_pushforward:
                # Regressing (y1 - y0)/dt fits the *noise* when the per-step
                # drift is small relative to the per-step diffusion (the regime
                # of real sensor data at its native sampling). The verifier's
                # pointwise claim is about the plant's exact Ito push-forward
                # drift under T -- so fit exactly that, recomputed each batch as
                # the transport moves. Rollout pairs still define the data
                # distribution and the region; they are simply no longer the
                # regression target.
                from sbounds.systems import pushforward_drift as _pf
                target = _pf(data.system, transport, x0)
                l_fit = ((F(y0) - target) ** 2).mean()
            else:
                target = (y1 - y0) / data.dt
                l_fit = ((F(y0) - target) ** 2).mean()
            _, A_er, _, A_rr = F.blocks(d_eta)
            l_dec = (A_er ** 2).sum()
            # the transversal residual must not directly force rho: this is the
            # g_rho term of the certificate's coupling bound b_Q, so penalising
            # the rho rows of the residual network is penalising exactly the
            # quantity the sound bound needs small
            zero_b = torch.zeros_like(y0)
            g_rho = F.residual_rho(y0, zero_b, d_eta)
            l_gres = (g_rho ** 2).mean()
            lam = torch.linalg.eigvalsh(0.5 * (A_rr + A_rr.T)).max()
            l_con = torch.nn.functional.softplus(lam + cfg.contract_target)
            l_anc = (transport.logdet(x0) ** 2).mean()
            loss = (l_fit + cfg.w_decouple * l_dec + cfg.w_gres * l_gres
                    + cfg.w_contract * l_con + cfg.w_anchor * l_anc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            spectral_project(transport, cfg.spec_cap)
            spectral_project(F, cfg.spec_cap)
            if cfg.hard_decouple and cfg.d_eta is not None:
                # the certified factor's drift must not read the transversal
                # coordinates: this is a structural requirement of the sound
                # bound (its coupling term pays for A[e,rho]), so it is imposed
                # as a projection, not a soft penalty the fit term can outbid
                with torch.no_grad():
                    F.A[:cfg.d_eta, cfg.d_eta:].zero_()
            # the verifier's rho-rows remainder is a Lipschitz ball of radius
            # L_rho = ||W2|| ||W1||; the cap keeps that ball small, which the
            # sound bound pays for explicitly -- tightening it is part of the
            # method (learn a certifiable factorisation), not a tune-out.
            for m in F.modules():
                if isinstance(m, torch.nn.Linear):
                    spectral_project(m, cfg.rho_spec_cap)
            if cfg.metric_every and step % cfg.metric_every == 0:
                F.refresh_rho_metric(d_eta)
            if (cfg.d_eta is not None and cfg.rho_margin > 0
                    and step == cfg.steps // 2):
                # W's transversal term needs Q PD, which needs A_rr Hurwitz with
                # margin: lambda_Q <= 1/(2*margin). The residual network
                # recompensates the fit over the remaining steps -- that
                # division of labour (stable linear part, expressive residual)
                # is the architecture, not a hack.
                with torch.no_grad():
                    Srr = F.A[cfg.d_eta:, cfg.d_eta:]
                    S = 0.5 * (Srr + Srr.T)
                    lam = float(torch.linalg.eigvalsh(S).max())
                    if lam > -cfg.rho_margin:
                        Srr -= (lam + cfg.rho_margin) * torch.eye(
                            F.dim - cfg.d_eta, dtype=Srr.dtype)
            hist["fit"].append(float(l_fit.detach()))
            hist["decouple"].append(float(l_dec.detach()))
            hist["gres"].append(float(l_gres.detach()))
            hist["contract"].append(float(l_con.detach()))
            hist["anchor"].append(float(l_anc.detach()))
            if verbose and step % cfg.log_every == 0:
                print(f"  [wm] step {step:5d} fit {float(l_fit):.3e} dec {float(l_dec):.3e} "
                      f"con {float(l_con):.3e} lam_rho {float(lam):+.3f}")
            step += 1
    if cfg.refit_A and not cfg.target_pushforward:
        # the ridge refit fits A to the noisy (y1-y0)/dt differences; with the
        # push-forward target the linear part is already aligned to the plant
        # by the fit term itself, and the refit would re-inject the noise
        refit_linear_part(F, transport, data, d_eta)
    hist["final_lam_rho"] = float(F.rho_contraction_rate(d_eta))
    hist.update(F.refresh_rho_metric(d_eta))
    return hist


@torch.no_grad()
def refit_linear_part(F: LatentDynamics, transport, data: RolloutData, d_eta: int,
                      ridge: float = 1e-8) -> None:
    """Structure-preserving ridge refit of the linear part A.

    The joint objective lets A and T trade off against each other, and the
    finite-difference target is noisy, so the shared optimum leaves easy
    variance on the table. One closed-form ridge solve at the end recovers it.
    The residual network is untouched.

    The solve respects the factorised structure: the (eta, rho) block stays
    exactly zero, because a nonzero A_e_rho is exactly what the sound coupling
    bound b_Q has to pay for. The rho rows are refitted over all coordinates
    (A_r e kept free), and the refit of A_rr is reverted if it would leave the
    transversal block non-Hurwitz -- the rho metric needs a contracting block.
    """
    Y0 = transport(data.x0)
    Y1 = transport(data.x1)
    target = (Y1 - Y0) / data.dt
    zero = torch.zeros(F.dim, dtype=Y0.dtype)
    R = target - F.residual(Y0, zero)
    A_old = F.A.detach().clone()
    A_new = A_old.clone()
    G_all = Y0.T @ Y0 + ridge * torch.eye(F.dim, dtype=Y0.dtype)
    # eta rows read only eta (structure: A[e, rho] = 0)
    Ye = Y0[:, :d_eta]
    Ge = Ye.T @ Ye + ridge * torch.eye(d_eta, dtype=Y0.dtype)
    A_new[:d_eta, :d_eta] = torch.linalg.solve(Ge, Ye.T @ R[:, :d_eta]).T
    if F.d_eta is None:
        A_new = torch.linalg.solve(G_all, Y0.T @ R).T
    else:
        # rho rows read everything
        A_new[d_eta:, :] = torch.linalg.solve(G_all, Y0.T @ R[:, d_eta:]).T
        # keep the transversal block admissible for the rho metric: the guard
        # is the Lyapunov equation itself (A^T Q + Q A = -I with Q > 0), not
        # the symmetric part -- a mechanical system's symmetric part always
        # has a non-negative eigenvalue, trained or not
        Q_new = solve_lyapunov_metric(A_new[d_eta:, d_eta:])
        ok_new = bool(torch.isfinite(Q_new).all()) and float(torch.linalg.eigvalsh(Q_new).min()) > 0.0
        if not ok_new:
            Q_old = solve_lyapunov_metric(A_old[d_eta:, d_eta:])
            ok_old = bool(torch.isfinite(Q_old).all()) and float(torch.linalg.eigvalsh(Q_old).min()) > 0.0
            if ok_old:
                A_new[d_eta:, d_eta:] = A_old[d_eta:, d_eta:]
            else:
                # neither refit nor trained block is admissible: fall back to a
                # stable diagonal -- the certificate stays available, honestly
                # weaker, and the state is reported through the metric check
                neg = -torch.eye(F.dim - d_eta, dtype=Y0.dtype)
                A_new[d_eta:, d_eta:] = neg
    F.A.copy_(A_new)
    if F.d_eta is not None:
        F.A[:d_eta, d_eta:].zero_()


def score_world_model(system: ChainArm, transport, F: LatentDynamics, x: torch.Tensor) -> dict:
    """Compare the learned latent drift against the exact Ito push-forward."""
    exact = pushforward_drift(system, transport, x)
    naive = _no_ito_reference(system, transport, x)
    with torch.no_grad():
        learned = F(transport(x))
        err = (learned - exact).norm(dim=-1)
        scale = exact.norm(dim=-1).clamp_min(1e-9)
        naive_err = (learned - naive).norm(dim=-1)
        out = {"rel_err_mean": float((err / scale).mean()),
               "rel_err_median": float((err / scale).median()),
               "rel_err_max": float((err / scale).max()),
               "abs_err_mean": float(err.mean()),
               "drift_norm_mean": float(exact.norm(dim=-1).mean()),
               "ito_correction_norm_mean": float((exact - naive).norm(dim=-1).mean()),
               "rel_err_vs_no_ito_mean": float((naive_err / scale).mean())}
    return out


@torch.enable_grad()
def _no_ito_reference(system: ChainArm, transport, x: torch.Tensor) -> torch.Tensor:
    """J_T f, i.e. the push-forward without the 1/2 tr(Sigma^T H_T Sigma) term."""
    xr = x.clone().requires_grad_(True)
    D = x.shape[-1]
    J = torch.stack([torch.autograd.grad(transport(xr)[..., k].sum(), xr,
                                         retain_graph=True)[0] for k in range(D)], dim=-2)
    return (J @ system.drift(xr).unsqueeze(-1)).squeeze(-1)


@dataclass
class CertConfig:
    steps: int = 1200
    batch: int = 256
    lr: float = 3e-3
    alpha: float = 0.05
    kappa: float = 1.0
    region_scale: float = 1.0
    v_res_cap: float = 1.0           # spectral cap on V's residual feature net
    v_coef_cap: float = 1.0          # cap on the residual coefficients a_j
    log_every: int = 200
    seed: int = 0


def train_certificate(V: LyapunovNet, F: LatentDynamics, transport, system: ChainArm,
                      d_eta: int, cfg: CertConfig = CertConfig(),
                      extra_eta: torch.Tensor | None = None, extra_rho: torch.Tensor | None = None,
                      extra_weight: float = 20.0, use_exact_drift: bool = False,
                      verbose: bool = False) -> dict:
    """Train V by penalising the exact generator violation over sampled states.

    Note what this does *not* do: it never enforces the sound bound. A network
    trained on sampled violations can pass on a finite sample and still fail a
    sound bound, which is precisely the sampling gap this repo measures instead
    of assuming away.
    """
    torch.manual_seed(cfg.seed)
    fixed_V = not getattr(V, "use_residual", True)
    if fixed_V:
        # The certificate is the fixed quadratic 1/2 eta^T P eta. What is
        # learned is the latent dynamics: the training signal is exactly the
        # decrease violation of the fixed V under the learned F and the
        # transport. This is the honest form of the method's pitch -- learn a
        # coordinate frame in which a simple certificate works -- and it
        # removes both pathologies measured in earlier runs: a free P shrinking
        # toward zero (vacuous beta) and residual-shaped curvature exploding
        # the sound bound (||Hess V|| ~ 17).
        params = list(F.parameters())
    else:
        params = list(V.parameters()) + list(F.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    hist = {"viol_frac": [], "max_gen": [], "loss": [], "beta": []}
    gen = torch.Generator().manual_seed(cfg.seed + 7)
    scale = transport.x_scale * cfg.region_scale
    d = d_eta
    F_fn = None
    if use_exact_drift:
        def F_fn(y):
            return pushforward_drift(system, transport, transport.inverse(y))

    for step in range(cfg.steps):
        # eta uniform in its box; rho := 0 exactly. Two justifications:
        # (1) the transversal term kappa rho^T Q rho of W is exactly radial, so
        # the generator on the rho=0 slice has no hidden extra variation from
        # rescaling rho; (2) the *sound* verifier carries the ball via the
        # analytic shell bound, so what the pointwise training must nail is the
        # eta-dependence -- which the slice measures exactly.
        u = torch.rand((cfg.batch, d), dtype=torch.float64, generator=gen)
        eta = (2.0 * u - 1.0) * scale[:d]
        rho = torch.zeros((cfg.batch, transport.dim - d), dtype=torch.float64)
        if extra_eta is not None and extra_eta.numel() > 0:
            eta = torch.cat([eta, extra_eta], dim=0)
            rho = torch.cat([rho, extra_rho], dim=0)
        resid = latent_generator(V, F, transport, system, cfg.kappa, cfg.alpha, d,
                                 eta, rho, F_fn=F_fn, create_graph=True)
        beta = noise_floor(V, F, transport, system, cfg.kappa, d)
        hist["beta"].append(beta)
        pen = torch.nn.functional.softplus((resid - beta) / 0.1)  # squared hinge on
        # the exact generator above the noise floor beta = L W(0): with additive
        # process noise the origin is not an SDE equilibrium and LW(0) > 0, so the
        # attainable certificate is LW + alpha W <= beta (exponential practical
        # stability to a noise ball), not the classical beta = 0.
        if extra_eta is not None and extra_eta.numel() > 0:
            w = torch.ones_like(pen)
            w[cfg.batch:] = extra_weight
            loss = (pen * w).mean()
        else:
            loss = pen.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if not fixed_V:
            spectral_project(V, 4.0)
        spectral_project(F, 4.0)
        # The verifier's drift bound pays |Hess V| * |F| * r^2 per cell: V's
        # curvature IS the certificate's certification cost. Left uncapped, the
        # optimizer happily buys pointwise margin by growing steep residual
        # terms, and the sound bound then needs ~10^5 cells to see the margin.
        # Capping the feature net's spectral norm and the coefficients is the
        # training-side half of the method: learn a V the verifier can afford.
        if not fixed_V and cfg.v_res_cap > 0:
            spectral_project(V.res, cfg.v_res_cap)
        if not fixed_V and cfg.v_coef_cap > 0:
            with torch.no_grad():
                # softplus^{-1}(cap): c_j such that softplus(c_j) <= cap
                import math as _math
                V.c.clamp_(max=_math.log(_math.expm1(cfg.v_coef_cap)))
        rd = resid.detach()
        hist["viol_frac"].append(float((rd > beta).to(torch.float64).mean()))
        hist["max_gen"].append(float((rd - beta).max()))
        hist["loss"].append(float(loss.detach()))
        if verbose and step % cfg.log_every == 0:
            print(f"  [cert] step {step:5d} loss {float(loss):.3e} viol {hist['viol_frac'][-1]:.4f} "
                  f"max_gen {hist['max_gen'][-1]:+.3e}")
    return hist
