"""Experiment 2: the gated online hot-swap.

The plant's damping degrades halfway through the stream (the `drift_system`
swap). A shadow pair retrains online on the fresh data and a candidate swap is
either accepted unconditionally (naive gate, what an unguarded online learner
does) or only when the sound interval bound certifies the candidate over the
region (sound gate). The measured claim is `unsafe_swaps`: swaps after which the
exact generator of the *real* plant is positive somewhere in the region.

Both gates run on the same data stream, same seeds, same drift change. Results
land in results/exp2.json.

Usage:  python experiments/exp2_shadow.py [--quick]
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, _jsonable
from sbounds.shadow import ShadowConfig, shadow_run
from sbounds.systems import ChainArm

CKPT = os.path.join(RESULTS, "exp1_ckpt")


def main() -> None:
    quick = "--quick" in sys.argv
    torch.set_num_threads(os.cpu_count() or 4)

    tag = "arm4_d2"
    ckpt = os.path.join(CKPT, f"{tag}.pt")
    if not os.path.exists(ckpt):
        print(f"missing checkpoint {ckpt}; run experiments/exp1_main.py first", flush=True)
        sys.exit(1)
    trained = torch.load(ckpt, weights_only=False)
    system, transport, F, V = (trained["system"], trained["transport"],
                               trained["F"], trained["V"])
    d_eta, kappa, alpha = trained["d_eta"], 1.0, 0.2

    # the plant drifts: damping drops by half (actuator degradation)
    drifted = ChainArm(n_links=trained["system"].n_links, sigma=trained["system"].sigma,
                       kd=trained["system"].kd * 0.5)
    region_lo, region_hi = trained["region"].init_box("full", scale=1.0)
    region_lo, region_hi = region_lo.squeeze(0), region_hi.squeeze(0)

    cfg_common = dict(rounds=8 if not quick else 6, batch=128, shadow_steps=60,
                      probe=384, patience=3, seed=0)
    out = {"drift": "kd halved at round rounds//2", "gates": {}}
    for gate in ("sound", "naive"):
        t0 = time.time()
        rep = shadow_run(system, V, F, transport, kappa, alpha, d_eta,
                         region_lo, region_hi, drift_system=drifted,
                         cfg=ShadowConfig(gate=gate, **cfg_common), verbose=True)
        dt = time.time() - t0
        out["gates"][gate] = {"accepted": rep.accepted, "rejected": rep.rejected,
                              "unsafe_swaps": rep.unsafe_swaps,
                              "unsafe_swap_fraction": rep.unsafe_swap_fraction,
                              "fallback_rounds": rep.fallback_rounds,
                              "rounds": rep.rounds, "seconds": dt}
        print(f"[{gate}] accepted={rep.accepted} rejected={rep.rejected} "
              f"unsafe_swaps={rep.unsafe_swaps} fallback={rep.fallback_rounds} "
              f"({dt:.0f}s)", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "exp2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=_jsonable)
    print("saved results/exp2.json", flush=True)


if __name__ == "__main__":
    main()
