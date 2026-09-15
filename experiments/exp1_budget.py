"""Equal-quality certification protocol for experiment 1.

The fixed-budget run (exp1_main) answers "who certifies at the same effort";
this driver answers "how much effort does each mode need". Per arm, the node
budget is doubled from BUDGET0 (4030) and BOTH modes re-certified at every
level, so the comparison is equal-effort at every rung. Factor mode stops
when it certifies >= TARGET of the region; full mode is capped at the same
level. The recorded ladder is itself the scientific claim: the effort factor
mode needs is set by the latent dimension d, while full mode does not finish
at any affordable budget once D is beyond ~8 (the curse of dimensionality).

Results are written after every level, so an interrupted run resumes by
re-running (levels are independent; each overwrites the ladder entry).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp1_main import CKPT, RESULTS, certify_both  # noqa: E402

BUDGET0 = 4030
CAP_BUDGET = BUDGET0 * 8      # 32k nodes: beyond this, factor mode's wall time
TARGET = 0.8                  # stops being the interesting quantity
SIZES = [(4, 2), (6, 3), (8, 4), (10, 4)]
TIME_BUDGET = 6 * 3600.0      # node budget is the binding constraint, not time


def _jsonable(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return o


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    out_path = os.path.join(RESULTS, "exp1_budget.json")
    final_path = os.path.join(RESULTS, "exp1.json")
    rows: list[dict] = []
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                rows = json.load(fh)["rows"]
        except (json.JSONDecodeError, KeyError):
            rows = []
    done_tags = {r["tag"] for r in rows}

    for n_links, d_eta in SIZES:
        tag = f"arm{n_links}_d{d_eta}"
        if tag in done_tags:
            print(f"[{tag}] already in ladder, skipping", flush=True)
            continue
        trained = torch.load(os.path.join(CKPT, f"{tag}.pt"), weights_only=False)
        ladder = []
        b = BUDGET0
        while True:
            t0 = time.time()
            c = certify_both(trained, b, TIME_BUDGET)
            ff = c["factor"]["certified_fraction"]
            fu = c["full"]["certified_fraction"]
            lvl = {"budget": b, "factor": c["factor"], "full": c["full"],
                   "seconds_factor": c["factor"]["seconds"],
                   "seconds_full": c["full"]["seconds"]}
            ladder.append(lvl)
            print(f"[{tag}] budget={b} factor={ff:.3f} ({c['factor']['nodes']} nodes) "
                  f"| full={fu:.3f} | {time.time() - t0:.0f}s", flush=True)
            row = {"tag": tag, "n_links": n_links, "D": trained["D"], "d_eta": d_eta,
                   "beta": c["beta"], "tol": c["tol"], "threshold": c["threshold"],
                   "lam_Q": trained["lam_Q"], "cert_viol_frac": trained["cert_viol_frac"],
                   "ladder": ladder}
            rows = [r for r in rows if r["tag"] != tag] + [row]
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"protocol": {"budget0": BUDGET0, "target": TARGET,
                                        "cap_budget": CAP_BUDGET}, "rows": rows},
                          fh, indent=2, default=_jsonable)
            if ff >= TARGET or b >= CAP_BUDGET:
                break
            b *= 2
        # merge the final level back into the canonical exp1.json schema so the
        # site and figures see the equal-quality numbers
        last = ladder[-1]
        if os.path.exists(final_path):
            with open(final_path, encoding="utf-8") as fh:
                fin = json.load(fh)
            for r in fin["rows"]:
                if r["D"] == trained["D"] and r["d_eta"] == d_eta:
                    r["factor"] = last["factor"]
                    r["full"] = last["full"]
                    r["budget"] = last["budget"]
                    r["budget_ladder"] = [{"budget": l["budget"],
                                           "factor": l["factor"]["certified_fraction"],
                                           "full": l["full"]["certified_fraction"]}
                                          for l in ladder]
                    break
            with open(final_path, "w", encoding="utf-8") as fh:
                json.dump(fin, fh, indent=2, default=_jsonable)
            print(f"[{tag}] merged final level into results/exp1.json", flush=True)
    print("saved results/exp1_budget.json", flush=True)


if __name__ == "__main__":
    main()
