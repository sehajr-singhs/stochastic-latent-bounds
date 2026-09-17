"""Figure generation from the experiment JSON files.

Pure-Python SVG writers: no matplotlib dependency, so figures can be
regenerated anywhere (including GitHub Actions) from the committed results.

Figures:
  fig1_tightness.svg    excess of the sound bound over the noise floor vs box
                        width, tight trace vs Cauchy-Schwarz (log-log)
  fig2_scaling.svg      certified fraction and node cost vs state dimension,
                        factor mode vs full mode
  fig3_shadow.svg       per-round acceptance/violation trace of the gated
                        hot-swap, sound gate vs naive gate
"""
from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")

INK = "#1a1a1a"
ACCENT = "#c62f2f"      # e3nn-style brick red
BLUE = "#33608c"
GREY = "#8a8a8a"


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _svg_open(w, h, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">',
        f"<title>{title}</title>",
    ]


def _text(x, y, s, size=12, fill=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def _line(x1, y1, x2, y2, stroke=GREY, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def _path(pts, stroke, width=2):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"/>'


def _dots(pts, fill, r=3):
    return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>' for x, y in pts)


def _axes(w, h, m, xlabel, ylabel, title):
    o = [_line(m, h - m, w - m, h - m, INK, 1.2), _line(m, m, m, h - m, INK, 1.2)]
    o.append(_text(w / 2, h - m / 2 + 2, xlabel, 13))
    o.append(_text(m - 34, h / 2, ylabel, 13).replace('text-anchor="middle"', 'transform="rotate(-90 %d %d)" text-anchor="middle"' % (m - 34, h / 2)))
    o.append(_text(w / 2, m / 2, title, 14, INK, "middle", "bold"))
    return o


def _xticks(m, w, h, lo, hi, ticks, fmt):
    o = []
    for t in ticks:
        x = m + (w - 2 * m) * (t - lo) / (hi - lo)
        o.append(_line(x, h - m, x, h - m + 5, INK, 1))
        o.append(_text(x, h - m + 20, fmt(t), 11))
    return o


def _yticks(m, w, h, lo, hi, ticks, fmt):
    o = []
    for t in ticks:
        y = (h - m) - (h - 2 * m) * (t - lo) / (hi - lo)
        o.append(_line(m - 5, y, m, y, INK, 1))
        o.append(_text(m - 9, y + 4, fmt(t), 11, INK, "end"))
    return o


# ---------------------------------------------------------------------------
# fig 1: tightness sweep
# ---------------------------------------------------------------------------

def fig_tightness():
    data = _load("exp3_tightness.json")
    if data is None:
        return False
    rows = data["rows"]
    w, h, m = 560, 380, 64
    xs = [r["width"] for r in rows]
    lo_x, hi_x = min(xs) * 0.7, max(xs) * 1.4
    lo_y, hi_y = 1e-4, max(max(r["cs_excess"] for r in rows),
                           max(r["tight_excess"] for r in rows)) * 2

    def X(v):
        return m + (w - 2 * m) * (v - lo_x) / (hi_x - lo_x)

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Bound excess over the noise floor vs box width (log-log)")
    o += _axes(w, h, m, "box half-width w", "sup(LW+aW) - beta", "")
    import math
    xt = [10 ** (math.floor(math.log10(lo_x)) + k)
          for k in range(int(math.log10(hi_x)) - int(math.log10(lo_x)) + 1)]
    o += _xticks(m, w, h, lo_x, hi_x, xt, lambda t: f"{t:g}")
    yt = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    o += _yticks(m, w, h, lo_y, hi_y, [t for t in yt if lo_y <= t <= hi_y],
                 lambda t: f"{t:g}")
    o.append(_path([(X(r["width"]), Y(max(r["cs_excess"], lo_y))) for r in rows], ACCENT))
    o.append(_dots([(X(r["width"]), Y(max(r["cs_excess"], lo_y))) for r in rows], ACCENT))
    o.append(_path([(X(r["width"]), Y(max(r["tight_excess"], lo_y))) for r in rows], BLUE))
    o.append(_dots([(X(r["width"]), Y(max(r["tight_excess"], lo_y))) for r in rows], BLUE))
    o.append(_text(w - m - 8, m + 14, "Cauchy-Schwarz Ito bound", 12, ACCENT, "end"))
    o.append(_text(w - m - 8, m + 32, "tight interval Ito trace", 12, BLUE, "end"))
    # tol line
    tol = data["tol"]
    o.append(_line(m, Y(tol), w - m, Y(tol), GREY, 1.2, "6 4"))
    o.append(_text(w - m - 8, Y(tol) - 5, f"tolerance tol = {tol:g}", 11, GREY, "end"))
    o.append("</svg>")
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "fig1_tightness.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


# ---------------------------------------------------------------------------
# fig 2: scaling
# ---------------------------------------------------------------------------

def fig_scaling():
    data = _load("exp1.json")
    if data is None:
        return False
    rows = data["rows"]
    w, h, m = 560, 380, 64
    dims = [r["D"] for r in rows]
    lo_x, hi_x = min(dims) - 1, max(dims) + 1
    lo_y, hi_y = -0.05, 1.05

    def X(v):
        return m + (w - 2 * m) * (v - lo_x) / (hi_x - lo_x)

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Certified fraction vs state dimension (same region, same budget)")
    o += _axes(w, h, m, "state dimension D", "certified fraction", "")
    o += _xticks(m, w, h, lo_x, hi_x, dims, lambda t: str(t))
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    fac = [(X(r["D"]), Y(r["factor"]["certified_fraction"])) for r in rows]
    full = [(X(r["D"]), Y(r["full"]["certified_fraction"])) for r in rows]
    o.append(_path(fac, BLUE))
    o.append(_dots(fac, BLUE))
    o.append(_path(full, ACCENT))
    o.append(_dots(full, ACCENT))
    o.append(_text(w - m - 8, m + 14, f"factor mode (d coords searched)", 12, BLUE, "end"))
    o.append(_text(w - m - 8, m + 32, "full mode (D coords searched)", 12, ACCENT, "end"))
    # node cost annotation under each pair
    for r in rows:
        o.append(_text(X(r["D"]), h - m + 34,
                       f"{r['factor']['nodes']}/{r['full']['nodes']} nodes", 9.5, GREY))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig2_scaling.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


# ---------------------------------------------------------------------------
# fig 3: shadow hot-swap trace
# ---------------------------------------------------------------------------

def fig_shadow():
    data = _load("exp2.json")
    if data is None:
        return False
    gates = data["gates"]
    w, h, m = 560, 380, 64
    n = max(len(g["rounds"]) for g in gates.values())
    lo_x, hi_x = -0.5, n - 0.5
    lo_y, hi_y = -0.05, 1.05

    def X(v):
        return m + (w - 2 * m) * (v - lo_x) / (hi_x - lo_x)

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Gated hot-swap: certified acceptance vs unconditional acceptance")
    o += _axes(w, h, m, "stream round", "probe violation fraction", "")
    o += _xticks(m, w, h, lo_x, hi_x, list(range(n)), lambda t: str(t))
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.5, 1.0], lambda t: f"{t:.1f}")
    o.append(_line(m, Y(0), w - m, Y(0), GREY, 1, "4 4"))
    o.append(_text(m + 6, Y(0) - 5, "safe: zero violations", 10.5, GREY))
    for name, color in (("sound", BLUE), ("naive", ACCENT)):
        g = gates[name]
        pts = [(X(r["round"]), Y(max(0.0, min(1.0, r["viol_frac_after"]))))
               for r in g["rounds"]]
        o.append(_path(pts, color))
        o.append(_dots(pts, color, 4))
    o.append(_text(w - m - 8, m + 14, "sound gate", 12, BLUE, "end"))
    o.append(_text(w - m - 8, m + 32, "naive gate", 12, ACCENT, "end"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig3_shadow.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def fig_realtrend():
    data = _load("exp4_realtrend.json")
    if data is None:
        return False
    era = data["era"]
    w, h, m = 560, 380, 64
    lo_y, hi_y = 0.0, 1.0

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Real aging drift: certificate violations under each fleet era")
    o += _axes(w, h, m, "", "probe violation fraction", "")
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    bars = [("healthy era", era["healthy"]["viol_frac"], BLUE, w * 0.32),
            ("aged era", era["aged"]["viol_frac"], ACCENT, w * 0.62)]
    bw = w * 0.14
    for label, v, color, cx in bars:
        y0, y1 = Y(0), Y(max(v, 0.005))
        o.append(f'<rect x="{cx - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                 f'height="{max(y0 - y1, 1.0):.1f}" fill="{color}" opacity="0.85"/>')
        o.append(_text(cx, y1 - 8, f"{v:.2f}", 13, color, weight="bold"))
        o.append(_text(cx, y0 + 16, label, 12, INK))
    shift = data["identification"]["attractor_shift"]
    o.append(_text(w - m - 8, m + 14, f"attractor shift {shift:.2f} sd", 11.5, GREY, "end"))
    o.append(_text(w - m - 8, m + 30,
                   f"identified from NASA C-MAPSS FD001", 11.5, GREY, "end"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig4_realtrend.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def fig_airquality():
    data = _load("exp5_airquality.json")
    if data is None:
        return False
    era = data["era"]
    w, h, m = 560, 380, 64
    lo_y, hi_y = 0.0, 1.0

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Seasonal drift: certificate violations under each heating regime")
    o += _axes(w, h, m, "", "probe violation fraction", "")
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    bars = [("non-heating era", era["healthy"]["viol_frac"], BLUE, w * 0.32),
            ("heating era", era["aged"]["viol_frac"], ACCENT, w * 0.62)]
    bw = w * 0.14
    for label, v, color, cx in bars:
        y0, y1 = Y(0), Y(max(v, 0.005))
        o.append(f'<rect x="{cx - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                 f'height="{max(y0 - y1, 1.0):.1f}" fill="{color}" opacity="0.85"/>')
        o.append(_text(cx, y1 - 8, f"{v:.2f}", 13, color, weight="bold"))
        o.append(_text(cx, y0 + 16, label, 12, INK))
    shift = data["identification"]["attractor_shift"]
    o.append(_text(w - m - 8, m + 14, f"attractor shift {shift:.2f} sd", 11.5, GREY, "end"))
    o.append(_text(w - m - 8, m + 30,
                   f"identified from Beijing PRSA (PM2.5 et al.)", 11.5, GREY, "end"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig5_airquality.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def fig_grid():
    data = _load("exp6_grid.json")
    if data is None:
        return False
    era = data["era"]
    w, h, m = 560, 380, 64
    lo_y, hi_y = 0.0, 1.0

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Load-regime drift: certificate violations under each grid era")
    o += _axes(w, h, m, "", "probe violation fraction", "")
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    bars = [("high-load era (08-20h)", era["high_load"]["viol_frac"], BLUE, w * 0.32),
            ("low-load era (21-07h)", era["low_load"]["viol_frac"], ACCENT, w * 0.62)]
    bw = w * 0.14
    for label, v, color, cx in bars:
        y0, y1 = Y(0), Y(max(v, 0.005))
        o.append(f'<rect x="{cx - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                 f'height="{max(y0 - y1, 1.0):.1f}" fill="{color}" opacity="0.85"/>')
        o.append(_text(cx, y1 - 8, f"{v:.2f}", 13, color, weight="bold"))
        o.append(_text(cx, y0 + 16, label, 12, INK))
    shift = data["identification"]["attractor_shift"]
    o.append(_text(w - m - 8, m + 14, f"attractor shift {shift:.2f} sd", 11.5, GREY, "end"))
    o.append(_text(w - m - 8, m + 30,
                   "identified from ETTm2 (transformer OT + 6 load channels)", 11.5, GREY, "end"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig6_grid.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def fig_baselines():
    data = _load("exp7_baselines.json")
    if data is None:
        return False
    domains = [("cmapss", "C-MAPSS"), ("aqi", "Air quality"), ("ett", "ETTm2 grid")]
    variants = [("pca", "PCA", BLUE), ("random", "Random proj.", GREY),
                ("learned", "Learned (ours)", ACCENT)]
    w, h, m = 640, 400, 64
    lo_y, hi_y = 0.0, 1.0

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Factor certified fraction: fixed linear maps vs the learned transport")
    o += _axes(w, h, m, "", "factor certified fraction", "")
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    group_w = (w - 2 * m) / len(domains)
    bw = group_w * 0.22
    for gi, (ds, label) in enumerate(domains):
        d = data.get(ds, {})
        cx0 = m + group_w * (gi + 0.5)
        for vi, (key, vlabel, color) in enumerate(variants):
            entry = d.get(key)
            frac = None
            if isinstance(entry, dict):
                cert = entry.get("cert") or {}
                frac = (cert.get("factor") or {}).get("certified_fraction")
            x = cx0 + (vi - 1) * (bw + 4)
            if frac is None:
                o.append(_text(x, Y(0) - 6, "n/a", 11, GREY))
                continue
            y0, y1 = Y(0), Y(max(frac, 0.005))
            o.append(f'<rect x="{x - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                     f'height="{max(y0 - y1, 1.0):.1f}" fill="{color}" opacity="0.85"/>')
            o.append(_text(x, y1 - 6, f"{frac:.2f}", 11.5, color, weight="bold"))
        o.append(_text(cx0, Y(0) + 16, label, 12, INK))
    for vi, (key, vlabel, color) in enumerate(variants):
        lx = m + 8 + vi * 130
        o.append(f'<rect x="{lx:.1f}" y="{m - 26}" width="10" height="10" fill="{color}" opacity="0.85"/>')
        o.append(_text(lx + 14, m - 17, vlabel, 11, INK, anchor="start"))
    o.append(_text(w - m - 8, m + 14,
                   "identical downstream protocol, budgets and seeds", 11, GREY, "end"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig7_baselines.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def fig_nonlinear():
    data = _load("exp9_nonlinear.json")
    if data is None:
        return False
    variants = [("pca", "PCA", BLUE), ("random", "Random proj.", GREY),
                ("learned", "Learned (ours)", ACCENT)]
    w, h, m = 640, 400, 64
    lo_y, hi_y = 0.0, 1.0

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Nonlinear N-link plant: fixed linear maps vs the learned transport")
    o += _axes(w, h, m, "", "factor certified fraction", "")
    o += _yticks(m, w, h, lo_y, hi_y, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}")
    group_w = (w - 2 * m)
    bw = group_w * 0.16
    cx0 = m + group_w * 0.5
    for vi, (key, vlabel, color) in enumerate(variants):
        entry = data.get(key)
        frac = None
        if isinstance(entry, dict):
            cert = entry.get("cert") or {}
            frac = (cert.get("factor") or {}).get("certified_fraction")
        x = cx0 + (vi - 1) * (bw + 26)
        if frac is None:
            o.append(_text(x, Y(0) - 6, "n/a", 11, GREY))
            continue
        y0, y1 = Y(0), Y(max(frac, 0.005))
        o.append(f'<rect x="{x - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                 f'height="{max(y0 - y1, 1.0):.1f}" fill="{color}" opacity="0.85"/>')
        o.append(_text(x, y1 - 6, f"{frac:.2f}", 11.5, color, weight="bold"))
        o.append(_text(x, Y(0) + 16, vlabel, 12, INK))
    o.append(_text(m + 8, m + 14, "N-link arm (D=8, d=2): the map must undo sin/cos dynamics", 11, GREY, "start"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig9_nonlinear.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def fig_rigor():
    data = _load("exp8_validation.json")
    if data is None or "ablations" not in data:
        return False
    ab = data["ablations"]
    w, h, m = 780, 320, 58
    lo_y, hi_y = 0.0, 1.0
    panel_w = (w - 2 * m) / 3

    def Y(v):
        return (h - m) - (h - 2 * m) * (v - lo_y) / (hi_y - lo_y)

    o = _svg_open(w, h, "Certificate sensitivity: kappa, noise scaling, region scaling")
    o.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
    panels = [("kappa (evaluation points)", [(p["kappa"], p["viol"]) for p in ab["kappa"]], "%.1f"),
              ("noise multiplier", [(p["noise_mult"], p["viol"]) for p in ab["noise"]], "x{:.1f}"),
              ("region scale", [(p["scale"], p["viol"]) for p in ab["region"]], "x{:.2f}")]
    for pi, (title, pts, fmt) in enumerate(panels):
        px = m + panel_w * pi
        pw = panel_w - 18
        # panel frame
        o.append(f'<line x1="{px:.1f}" y1="{Y(1):.1f}" x2="{px:.1f}" y2="{Y(0):.1f}" stroke="{GREY}"/>')
        o.append(f'<line x1="{px:.1f}" y1="{Y(0):.1f}" x2="{px + pw:.1f}" y2="{Y(0):.1f}" stroke="{GREY}"/>')
        xs = [p[0] for p in pts]
        x_lo, x_hi = min(xs), max(xs)

        def X(t):
            return px + 6 + (pw - 12) * (t - x_lo) / max(x_hi - x_lo, 1e-9)

        path = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
        o.append(f'<polyline points="{path}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
        for t, v in pts:
            o.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{ACCENT}"/>')
            o.append(_text(X(t), Y(0) + 16, fmt.format(t), 10, GREY))
            o.append(_text(X(t), Y(v) - 8, f"{v:.2f}", 10, INK))
        o.append(_text(px + pw / 2, m - 22, title, 11.5, INK, weight="bold"))
    o.append(_text(m, h - 12, "pointwise violation fraction vs sweep parameter "
                   "(C-MAPSS checkpoint, 512 probes)", 10.5, GREY, "start"))
    o.append("</svg>")
    with open(os.path.join(SITE, "fig8_rigor.svg"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    return True


def main():
    done = []
    for fn, name in ((fig_tightness, "fig1_tightness"),
                     (fig_scaling, "fig2_scaling"),
                     (fig_shadow, "fig3_shadow"),
                     (fig_realtrend, "fig4_realtrend"),
                     (fig_airquality, "fig5_airquality"),
                     (fig_grid, "fig6_grid"),
                     (fig_baselines, "fig7_baselines"),
                     (fig_rigor, "fig8_rigor")):
        ok = fn()
        print(("wrote " if ok else "skipped (no data) ") + name)
        if ok:
            done.append(name)
    return done


if __name__ == "__main__":
    main()
