"""Stage-by-stage probe of the fixed quadratic-V certificate mode."""
import sys, time, math, torch
sys.path.insert(0, '.')
from sbounds.systems import ChainArm
from sbounds.transport import InvertibleTransport
from sbounds.models import LyapunovNet, LatentDynamics
from sbounds.train import TrainConfig, CertConfig, generate_pairs, train_world_model, train_certificate
from sbounds.generator import noise_floor
from experiments.exp1_main import ALPHA, KAPPA

torch.manual_seed(0)
n_links, d_eta = 4, 2
t0 = time.time()
system = ChainArm(n_links=n_links, sigma=0.05); D = system.dim
data = generate_pairs(system, n_traj=800, dt=0.01, seed=0)
transport = InvertibleTransport(dim=D, d_latent=d_eta, n_layers=4, width=32, hidden_depth=2,
    x_star=system.equilibrium(), x_scale=system.lqr_like_scale(), seed=0)
F = LatentDynamics(D, width=64, depth=2, seed=0, d_eta=d_eta)
print(f'{time.time()-t0:.1f}s data+build', flush=True)
train_world_model(transport, F, data, d_eta, TrainConfig(steps=500, seed=0, w_contract=0.3, d_eta=d_eta, rho_spec_cap=1.0))
print(f'{time.time()-t0:.1f}s world model', flush=True)
V = LyapunovNet(d_eta, use_residual=False, p_scale=1.0, seed=0)
beta = noise_floor(V, F, transport, system, KAPPA, d_eta)
print(f'{time.time()-t0:.1f}s beta BEFORE cert training = {beta:.4f}', flush=True)
ch = train_certificate(V, F, transport, system, d_eta, CertConfig(steps=600, alpha=ALPHA, kappa=KAPPA, seed=0), verbose=False)
print(f'{time.time()-t0:.1f}s cert training: viol_frac={ch["viol_frac"][-1]:.4f} max_gen={ch["max_gen"][-1]:+.4f}', flush=True)
beta = noise_floor(V, F, transport, system, KAPPA, d_eta)
import numpy as np
lamQ = float(np.linalg.eigvalsh(F.rho_metric(d_eta).numpy()).max())
print(f'beta after = {beta:.4f} lam_Q = {lamQ:.3f}', flush=True)
torch.save({'V': V, 'F': F, 'transport': transport, 'system': system, 'region_dict': {
    'eta_scale': None, 'rho_radius': None}}, '/tmp/fixedv_stage.pt')
print('DONE', flush=True)
