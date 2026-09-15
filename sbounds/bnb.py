"""Worst-first branch and bound over a box region, with an explicit node budget.

The correctness contract is one-sided on purpose: a box is *certified* only when
its sound upper bound on sup(L W + alpha W) is strictly negative. Boxes that are
merely "not disproved" are counted as unknown, never as certified, and the
returned certified fraction is always a lower bound on the true one.

Reporting the node count next to the certified fraction is the point of the
whole exercise: it is the currency in which the curse of dimensionality is paid.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import torch


@dataclass
class BnBResult:
    certified_fraction: float
    nodes: int
    seconds: float
    unknown_boxes: torch.Tensor | None      # (K, D) lower corners of unresolved boxes
    unknown_boxes_hi: torch.Tensor | None
    worst_upper: float
    fully_certified: bool


def box_measure(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Volume of each box, computed in log space to stay finite in high dimension."""
    return torch.exp(torch.log((hi - lo).clamp_min(0.0)).sum(-1))


def worst_first_bnb(bound_fn, lo0: torch.Tensor, hi0: torch.Tensor,
                    split_dims: torch.Tensor, node_budget: int = 20000,
                    batch: int = 32, time_budget: float | None = None,
                    return_unknown: bool = True, threshold: float = 0.0) -> BnBResult:
    """Certify as much of [lo0, hi0] as the node budget allows.

    bound_fn(lo, hi) -> (B,) sound upper bounds on sup(L W + alpha W) per box.
    split_dims indexes the coordinates that BnB is allowed to subdivide.
    threshold is the certificate constant beta: a box counts as certified when
    its sound upper bound is <= beta. With additive process noise beta = L W(0)
    > 0 (the noise floor), and the classical beta = 0 case is the special case
    threshold=0.
    """
    t0 = time.time()
    split_dims = split_dims.to(torch.long)
    total = float(box_measure(lo0, hi0).sum())
    lo0, hi0 = lo0.clone(), hi0.clone()
    # a single start box may arrive with a leading batch dim (1, D); every box
    # inside the search is a (D,) vector, children stack to (B, D)
    if lo0.dim() > 1 and lo0.shape[0] == 1:
        lo0, hi0 = lo0.squeeze(0), hi0.squeeze(0)

    up0 = bound_fn(lo0, hi0).reshape(-1)
    if bool((up0 <= threshold).all()):
        return BnBResult(1.0, 0, 0.0, None, None, float(up0.max()), True)

    # heap entries: (-worst upper in the box, tie-break, unique id, lo, hi, upper)
    heap: list = []
    uid = 0
    heapq.heappush(heap, (-float(up0.max()), 0, uid, lo0, hi0, up0))
    certified_vol = 0.0
    nodes = 0
    worst = float(up0.max())

    while heap and nodes < node_budget:
        if time_budget is not None and (time.time() - t0) > time_budget:
            break
        take = min(batch, len(heap))
        popped = [heapq.heappop(heap) for _ in range(take)]
        c_lo, c_hi = [], []
        for _, _, _, lo, hi, _ in popped:
            w = (hi - lo)[..., split_dims]
            k = int(torch.argmax(w))
            d = int(split_dims[k])
            mid = 0.5 * (lo[..., d] + hi[..., d])
            left_lo, left_hi = lo.clone(), hi.clone()
            left_hi[..., d] = mid
            right_lo, right_hi = lo.clone(), hi.clone()
            right_lo[..., d] = mid
            c_lo += [left_lo, right_lo]
            c_hi += [left_hi, right_hi]
        c_lo = torch.stack(c_lo)
        c_hi = torch.stack(c_hi)
        up = bound_fn(c_lo, c_hi).reshape(-1)
        nodes += c_lo.shape[0]
        ok = up <= threshold
        if bool(ok.any()):
            certified_vol += float(box_measure(c_lo[ok], c_hi[ok]).sum())
        for i in torch.nonzero(~ok, as_tuple=False).reshape(-1).tolist():
            uid += 1
            heapq.heappush(heap, (-float(up[i]), 0, uid, c_lo[i], c_hi[i], up[i]))

    if heap:
        worst = max(float(e[5].max()) for e in heap)
    unknown_lo = unknown_hi = None
    if return_unknown and heap:
        unknown_lo = torch.stack([e[3] for e in heap])
        unknown_hi = torch.stack([e[4] for e in heap])
    return BnBResult(
        certified_fraction=certified_vol / total if total > 0 else 1.0,
        nodes=nodes,
        seconds=time.time() - t0,
        unknown_boxes=unknown_lo,
        unknown_boxes_hi=unknown_hi,
        worst_upper=worst,
        fully_certified=not heap,
    )


def bisect_radius(bound_fn_factory, lo0: torch.Tensor, hi0: torch.Tensor,
                  split_dims: torch.Tensor, node_budget: int = 20000,
                  quantile: float = 0.999, iters: int = 6,
                  lo_scale: float = 1e-3, hi_scale: float = 8.0,
                  time_budget: float | None = None, threshold: float = 0.0) -> dict:
    """Largest radial scale for which the certified fraction reaches `quantile`.

    bound_fn_factory(scale) -> bound_fn, so each candidate scale can rebuild
    whatever cached quantities it needs.
    """
    best = {"scale": 0.0, "certified_fraction": 0.0, "nodes": 0, "seconds": 0.0}
    lo_s, hi_s = lo_scale, hi_scale
    trace = []
    for _ in range(iters):
        mid = 0.5 * (lo_s + hi_s)
        res = worst_first_bnb(bound_fn_factory(mid), lo0 * mid, hi0 * mid, split_dims,
                              node_budget=node_budget, time_budget=time_budget,
                              return_unknown=False, threshold=threshold)
        trace.append({"scale": mid, "certified_fraction": res.certified_fraction,
                      "nodes": res.nodes, "seconds": res.seconds})
        if res.certified_fraction >= quantile:
            best = {"scale": mid, "certified_fraction": res.certified_fraction,
                    "nodes": res.nodes, "seconds": res.seconds}
            lo_s = mid
        else:
            hi_s = mid
    best["trace"] = trace
    return best
