"""arm4 decomposition: where does the factor bound's 349 come from?"""
import sys, time, os
sys.path.insert(0, '.')
import torch
from sbounds.bounds import bound_factor
from sbounds.generator import noise_floor, latent_generator
from experiments.exp1_main import build_and_train, ALPHA, KAPPA

torch.set_num_threads(8)
CKPT = "results/arm4_cap.pt"
t0 = time.time()
if os.path.exists(CKPT):
    t = torch.load(CKPT, weights_only=False)
else:
    t = build_and_train(n_links=4, d_eta=2, sigma=0.05, alpha=ALPHA, kappa=KAPPA, seed=0,
                        wm_steps=1200, cert_steps=1500, n_traj=800)
    torch.save(t, CKPT)
V, F, T, S, region = t['V'], t['F'], t['transport'], t['system'], t['region']
d = t['d_eta']
beta = noise_floor(V, F, T, S, KAPPA, d)
with torch.no_grad():
    P = V.P
print('beta=%.6f  ||P||=%.4f  P_min_eig=%.5f  coef_max=%.4f' % (
    beta, float(torch.linalg.norm(P)), float(torch.linalg.eigvalsh(P).min()),
    float(torch.nn.functional.softplus(V.c).max())), flush=True)
# pointwise generator over the region INCLUDING rho != 0
g = torch.Generator().manual_seed(5)
n = 40000
u = torch.rand((n, d), dtype=torch.float64, generator=g)
eta = (2*u - 1) * region.eta_scale
v = torch.randn((n, T.dim - d), dtype=torch.float64, generator=g)
v = v / v.norm(dim=-1, keepdim=True)
rad = region.rho_radius * torch.rand((n,1), dtype=torch.float64, generator=g) ** (1.0/(T.dim-d))
rho = v * rad
res = latent_generator(V, F, T, S, KAPPA, ALPHA, d, eta, rho)
print('POINTWISE (eta box x rho ball): max=%.4f p99=%.4f frac>thr=%.5f' % (
    float(res.max()), float(res.quantile(0.99)), float((res > beta+0.05).to(torch.float64).mean())), flush=True)
# decomposition at worst-cell sizes, corner cell, outer shell vs inner
lo_c = -region.eta_scale/2; hi_c = torch.zeros(d, dtype=torch.float64)  # corner sub-cell
for w in (region.eta_scale[0].item(), 0.25, 0.05, 0.01):
    lo = -torch.full((1,d), w, dtype=torch.float64); hi = torch.full((1,d), w, dtype=torch.float64)
    b = bound_factor(V, F, T, S, KAPPA, ALPHA, d, lo, hi, region.rho_radius, rho_rings=8)
    b_in = bound_factor(V, F, T, S, KAPPA, ALPHA, d, lo, hi, region.rho_radius/8, rho_rings=1)
    print('w=%.3f full-ball: up=%9.3f de=%9.3f dr=%8.3f ito=%8.4f aW=%8.3f | inner-ball: up=%8.3f' % (
        w, float(b.upper[0]), float(b.drift_eta[0]), float(b.drift_rho[0]),
        float(b.ito[0]), float(b.upper[0])-float(b.drift_eta[0])-float(b.drift_rho[0])-float(b.ito[0]),
        float(b_in.upper[0])), flush=True)
print('DONE %.0fs' % (time.time()-t0), flush=True)
