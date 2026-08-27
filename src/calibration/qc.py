"""Annotation QC: monotonicity checks, leave-one-flag-out CV, cross-photo drift."""
import numpy as np

from src.calibration.data import PhotoData, flag_pixels, markers
from src.calibration.ground_plane import GroundPlaneFit


def _must_decrease(pairs, transect, kind, fmt):
    """Violations where the value in (dist, value) pairs fails to strictly
    decrease with distance."""
    pairs = sorted(pairs)
    return [{"transect": transect, "kind": kind, "dist_a": d1, "dist_b": d2,
             "detail": fmt.format(a, b)}
            for (d1, a), (d2, b) in zip(pairs, pairs[1:]) if b >= a]


def monotonicity(photo):
    pix = flag_pixels(photo)
    violations = []
    for name in "LCR":
        # ground v must strictly decrease with distance
        violations += _must_decrease([(d, v) for (d, t), (_, v) in pix.items() if t == name],
                                     name, "ground_v", "v {:.0f} -> {:.0f} (must decrease)")
        # apparent scale (px per cm) must strictly decrease with distance
        scale = {}
        for s in photo.size:
            if s.transect == name:
                scale.setdefault(s.dist, []).append(s.px_len / s.cm_len)
        violations += _must_decrease([(d, float(np.median(v))) for d, v in scale.items()],
                                     name, "size_scale", "px/cm {:.2f} -> {:.2f} (must decrease)")
    return violations


def _photo_without_flag(photo, dist, transect):
    keep = lambda o: not (o.dist == dist and o.transect == transect)
    return PhotoData(photo.site, photo.image, photo.image_w, photo.image_h,
                     [g for g in photo.ground if keep(g)], [s for s in photo.size if keep(s)],
                     photo.skipped)


def leave_one_flag_out(photo):
    rows = []
    for (dist, transect), obs in sorted(markers(photo).items()):
        u, v = float(np.mean([g.u for g in obs])), float(np.mean([g.v for g in obs]))
        held_out = _photo_without_flag(photo, dist, transect)
        pb, sb = GroundPlaneFit.fit(held_out).predict(u, v)
        rows.append({"site": photo.site, "image": photo.image,
                     "transect": transect, "dist": dist, "n_obs": len(obs),
                     "pred_b": pb, "err_b": None if pb is None else pb - dist})
    return rows


def cross_photo(models):
    """models: dict site -> list of fitted GroundPlaneFit (one per photo of that camera)."""
    rows = []
    for site, ms in sorted(models.items()):
        ok = [m for m in ms if m.ok]
        if len(ok) < 2:
            continue
        fs = [m.params.f for m in ok]
        hs = [m.params.h for m in ok]
        rows.append({"site": site,
                     "f_spread_pct": 100.0 * (max(fs) - min(fs)) / min(fs),
                     "h_spread_m": max(hs) - min(hs)})
    return rows
