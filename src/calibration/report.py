"""QC outputs: CSVs, per-photo overlay plots, markdown summary."""
import csv
from pathlib import Path

import numpy as np

from src.calibration.ground_plane import pixel_to_distance


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def overlay(photo, plane_fit, img_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.imshow(Image.open(img_path))
    if plane_fit.ok:
        us = np.arange(0, photo.image_w, 8)
        vs = np.arange(0, photo.image_h, 4)
        uu, vv = np.meshgrid(us, vs)
        dd = pixel_to_distance(uu.ravel(), vv.ravel(), plane_fit.params,
                               plane_fit.cx, plane_fit.cy).reshape(uu.shape)
        cs = ax.contour(uu, vv, dd, levels=range(2, 16), linewidths=0.7, cmap="cool")
        ax.clabel(cs, fmt="%d m", fontsize=7)
    sc = ax.scatter([g.u for g in photo.ground], [g.v for g in photo.ground],
                    c=[g.dist for g in photo.ground], cmap="autumn", s=18,
                    edgecolors="black", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="labeled distance (m)")
    ax.set_title(f"{photo.site} / {photo.image}")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _abs_errs(cv_rows, key):
    return sorted(abs(r[key]) for r in cv_rows if r[key] is not None)


def _fmt_quantiles(errs):
    if not errs:
        return "n/a"
    percentile = lambda p: errs[min(len(errs) - 1, int(p * len(errs)))]
    return f"median {percentile(0.5):.2f} m, p90 {percentile(0.9):.2f} m, max {errs[-1]:.2f} m (n={len(errs)})"


def summarize(cv_rows, mono, skipped, cross, insufficient):
    lines = ["# Calibration QC summary", ""]
    lines.append(f"- Photos with insufficient data for the ground-plane fit: {insufficient or 'none'}")
    lines.append(f"- Monotonicity violations: {len(mono)} (see monotonicity.csv)")
    lines.append(f"- Vertical spans skipped for extreme lean: {len(skipped)} (see skipped_leans.csv)")
    lines.append("")
    lines.append("## Leave-one-flag-out CV, absolute error")
    lines.append(f"- Ground-plane fit:  {_fmt_quantiles(_abs_errs(cv_rows, 'err_b'))}")
    lines.append("")
    lines.append("### By distance band")
    for lo, hi in [(0, 5), (5, 10), (10, 16)]:
        band = [r for r in cv_rows if lo < r["dist"] <= hi]
        lines.append(f"- {lo}-{hi} m: {_fmt_quantiles(_abs_errs(band, 'err_b'))}")
    lines.append("")
    lines.append("## Worst 20 flags by leave-one-flag-out error (annotation suspects)")
    worst = sorted((r for r in cv_rows if r["err_b"] is not None),
                   key=lambda r: -abs(r["err_b"]))[:20]
    lines.append("| site | image | transect | dist | err_b (m) |")
    lines.append("|---|---|---|---|---|")
    for r in worst:
        lines.append(f"| {r['site']} | {r['image']} | {r['transect']} | {r['dist']} "
                     f"| {r['err_b']:+.2f} |")
    lines.append("")
    lines.append("## Cross-photo drift (fitted camera constants per site)")
    lines.append("| site | f spread % | h spread m |")
    lines.append("|---|---|---|")
    for r in cross:
        lines.append(f"| {r['site']} | {r['f_spread_pct']:.1f} | {r['h_spread_m']:.2f} |")
    return "\n".join(lines) + "\n"
