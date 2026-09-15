"""stochastic-latent-bounds: certified stochastic Lyapunov bounds in a learned
low-dimensional latent factor of a physical system.

Pipeline, end to end:

    systems.ChainArm            analytic physical plant + process noise
    train.generate_pairs        rollout pairs, the only training signal
    transport.InvertibleTransport  learned diffeomorphism, T(x*) = 0
    models.LatentDynamics       latent SDE drift F, F(0) = 0 by construction
    generator.latent_generator  exact L W + alpha W (training, counterexamples)
    bounds.bound_factor/full    sound interval bounds on sup(L W + alpha W)
    bnb.worst_first_bnb         worst-first branch and bound under a node budget
    cegis.cegis_loop            verifier-in-the-loop retraining
    shadow.shadow_run           gated online hot-swap with fallback

Read `docs/math.md` for the derivations and `docs/results.md` for the measured
numbers, including the negative results.
"""

__version__ = "0.1.0"

from .bounds import BoundResult, bound_factor, bound_full, sigma_frobenius_bound
from .bnb import BnBResult, bisect_radius, worst_first_bnb
from .generator import latent_generator
from .models import LatentDynamics, LyapunovNet, spectral_project
from .systems import ChainArm, pushforward_drift
from .transport import InvertibleTransport
from .train import RolloutData, euler_maruyama, generate_pairs, train_certificate, train_world_model

__all__ = [
    "BoundResult", "bound_factor", "bound_full", "sigma_frobenius_bound",
    "BnBResult", "bisect_radius", "worst_first_bnb", "latent_generator",
    "LatentDynamics", "LyapunovNet", "spectral_project", "ChainArm",
    "pushforward_drift", "InvertibleTransport", "RolloutData", "euler_maruyama",
    "generate_pairs", "train_certificate", "train_world_model",
    "__version__",
]
