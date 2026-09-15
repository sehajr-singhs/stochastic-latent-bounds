"""End-to-end validation of the cascade fix: arm4, then certify factor vs full.

The claim under test: with res_rho reading only eta, the coupling bound b_Q
scales with the eta box, the rho term shrinks under subdivision, and factor
mode certifies a real fraction of the region at a small node budget.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from sbounds.bounds import bound_factor
from sbounds.generator import noise_floor, latent_generator
from experiments.exp1_main import build_and_train, ALPHA, KAPPA, TOL
from sbounds.region import make_bound_fn, Region
from sbounds.bnb import worst_first_bnb

torch.set_num_threads(max(os.cpu_count() or 4, 4))
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cascade_arm4.pt")
t0 = time.time()
if os.path.exists(CKPT):
    t = torch.load(CKPT, weights_only=False)
else:
    t = build_and_train(n_links=4, d_eta=2, sigma=0.05, alpha=ALPHA, kappa=KAPPA, seed=0,
                        wm_steps=600, cert_steps=600, n_traj=800)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save(t, CKPT)
V, F, T, S, region = t['V'], t['F'], t['transport'], t['system'], t['region']
d = t['d_eta']
beta = noise_floor(V, F, T, S, KAPPA, d)
print('beta=%.6f lam_Q=%.4f rho_r=%.3f eta_scale=%s viol_frac=%.3f' % (
    beta, t['lam_Q'], region.rho_radius, [round(float(v), 3) for v in region.eta_scale],
    t['cert_viol_frac']), flush=True)

# pointwise residual over the region (eta box x rho ball)
g = torch.Generator().manual_seed(5)
n = 20000
u = torch.rand((n, d), dtype=torch.float64, generator=g)
eta = (2 * u - 1) * region.eta_scale
v = torch.randn((n, T.dim - d), dtype=torch.float64, generator=g)
v = v / v.norm(dim=-1, keepdim=True)
rad = region.rho_radius * torch.rand((n, 1), dtype=torch.float64, generator=g) ** (1.0 / (T.dim - d))
rho = v * rad
res = latent_generator(V, F, T, S, KAPPA, ALPHA, d, eta, rho)
thr = beta + TOL
print('POINTWISE: max=%.4f frac>thr=%.5f' % (float(res.max()), float((res > thr).to(torch.float64).mean())), flush=True)

# decomposition on representative cells
for w in (float(region.eta_scale[0]), 0.1, 0.025):
    lo = -torch.full((1, d), w, dtype=torch.float64)
    hi = torch.full((1, d), w, dtype=torch.float64)
    r = region.rho_radius
    outs = []
    for k in range(8):
        r_out, r_in = r * (k + 1) / 8, r * k / 8
        b = bound_factor(V, F, T, S, KAPPA, ALPHA, d, lo, hi, r_out, r_in=r_in)
        outs.append((k, float(b.upper[0]), float(b.drift_eta[0]), float(b.drift_rho[0]), float(b.ito[0])))
    worst = max(o[1] for o in outs)
    print('w=%.3f worst_shell=%.4f  (shell: up, de, dr, ito)' % (w, worst), flush=True)
    for k, up, de, dr, it in outs:
        print('   shell%d up=%9.4f de=%9.4f dr=%9.4f ito=%8.5f' % (k, up, de, dr, it), flush=True)
    if worst <= thr:
        print('   -> certifies at w=%.3f' % w, flush=True)

# full BnB on the region, factor mode
bf = make_bound_fn('factor', V, F, T, S, KAPPA, ALPHA, region, rho_rings=8)
lo, hi = region.init_box('factor')
res_bnb = worst_first_bnb(bf, lo, hi, region.split_dims('factor'), node_budget=4000,
                          time_budget=1800.0, return_unknown=False, threshold=thr)
print('BnB factor: frac=%.4f nodes=%d secs=%.0f worst=%.4f full_cert=%s' % (
    res_bnb.certified_fraction, res_bnb.nodes, res_bnb.seconds, res_bnb.worst_upper,
    res_bnb.fully_certified), flush=True)
print('DONE %.0fs' % (time.time() - t0), flush=True)
