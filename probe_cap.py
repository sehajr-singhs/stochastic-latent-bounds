"""arm4 probe v2: real BnB, both modes, near-quadratic certificate."""
import sys, time
sys.path.insert(0, '.')
import torch
from sbounds.generator import noise_floor
from sbounds.region import make_bound_fn
from sbounds.bnb import worst_first_bnb
from experiments.exp1_main import build_and_train, certify_both, ALPHA, KAPPA

torch.set_num_threads(8)
t0 = time.time()
t = build_and_train(n_links=4, d_eta=2, sigma=0.05, alpha=ALPHA, kappa=KAPPA, seed=0,
                    wm_steps=1200, cert_steps=1500, n_traj=800)
d = t['d_eta']
beta = noise_floor(t['V'], t['F'], t['transport'], t['system'], KAPPA, d)
print('beta=%.5f lam_Q=%.3f wm=%.3f viol=%.4f [%.0fs]' % (
    beta, float(torch.linalg.eigvalsh(t['F'].rho_metric(d)).max()),
    t['world_model']['rel_err_mean'], t['cert_viol_frac'], time.time()-t0), flush=True)
out = certify_both(t, node_budget=30000, time_budget=600.0, chunk=96)
for mode in ("factor", "full"):
    r = out[mode]
    print('%-6s frac=%.4f nodes=%d secs=%.0f worst=%.4f' % (
        mode, r['certified_fraction'], r['nodes'], r['seconds'], r['worst_upper']), flush=True)
print('DONE %.0fs' % (time.time()-t0), flush=True)
