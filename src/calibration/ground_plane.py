"""Pinhole camera over a flat ground plane — the 4-parameter calibration.

Camera frame: +x right, +y down, +z forward (optical axis).
World frame: camera at origin, same axes with pitch/roll removed;
ground plane at y = +h. pitch > 0 tilts the optical axis downward.
"""
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.calibration.data import CALIB_DIR

FLAG_BODY_CENTER_HEIGHT_M = 0.46  # flag-body center height above ground, m (49.53 - 6.35/2 cm)


@dataclass
class PlaneParams:
    f: float      # focal length, px
    h: float      # camera height above ground plane, m
    pitch: float  # rad, >0 = camera looks down
    roll: float   # rad


def _cam_to_world(xc, yc, zc, p):
    cr, sr = math.cos(p.roll), math.sin(p.roll)
    xr = xc * cr - yc * sr
    yr = xc * sr + yc * cr
    ct, st = math.cos(p.pitch), math.sin(p.pitch)
    yw = yr * ct + zc * st
    zw = -yr * st + zc * ct
    return xr, yw, zw


def _world_to_cam(xw, yw, zw, p):
    ct, st = math.cos(p.pitch), math.sin(p.pitch)
    yr = yw * ct - zw * st
    zc = yw * st + zw * ct
    cr, sr = math.cos(p.roll), math.sin(p.roll)
    xc = xw * cr + yr * sr
    yc = -xw * sr + yr * cr
    return xc, yc, zc


def pixel_to_distance(u, v, p, cx, cy):
    """Horizontal ground distance (m) for pixel rays; nan where no ground hit."""
    xc = (np.asarray(u, float) - cx) / p.f
    yc = (np.asarray(v, float) - cy) / p.f
    xw, yw, zw = _cam_to_world(xc, yc, np.ones_like(xc), p)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((yw > 1e-9) & (zw > 0), p.h / yw, np.nan)
    return t * np.hypot(xw, zw)


def world_to_pixel(x, y, z, p, cx, cy):
    xc, yc, zc = _world_to_cam(x, y, z, p)
    return cx + p.f * xc / zc, cy + p.f * yc / zc


def flag_span_px(dist_m, cm_len, p, vertical):
    """Apparent length (px) of a flag span at horizontal ground distance dist_m.
    Vectorised over dist_m / cm_len / vertical."""
    L = np.asarray(cm_len, float) / 100.0
    dz = p.h - FLAG_BODY_CENTER_HEIGHT_M          # camera height above flag-body center
    r2 = np.asarray(dist_m, float) ** 2 + dz ** 2   # squared slant range to the flag body
    # vertical stick seen at depression angle: perpendicular component = L * D / r;
    # a horizontal span is already perpendicular to the ray.
    return np.where(vertical, p.f * L * dist_m / r2, p.f * L / np.sqrt(r2))


def dense_distance_map(p, cx, cy, homography=None, step=4, width=1920, height=1080,
                       d_min=2.0, d_max=18.0):
    """Horizontal ground distance on a `step`-pixel grid over a width x height
    frame, NaN outside [d_min, d_max]. With `homography` (3x3, target -> the
    calibrated frame), grid points are mapped through it first, so the map is
    expressed in the target's frame."""
    us, vs = np.meshgrid(np.arange(0, width, step, dtype=float),
                         np.arange(0, height, step, dtype=float))
    if homography is not None:
        pts = homography @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
        us, vs = (pts[0] / pts[2]).reshape(us.shape), (pts[1] / pts[2]).reshape(vs.shape)
    d = pixel_to_distance(us, vs, p, cx, cy)
    return np.where((d >= d_min) & (d <= d_max), d, np.nan)


MIN_GROUND_OBS = 6
MIN_DISTINCT_DISTS = 3
SIZE_TERM_WEIGHT = 1.0  # relative weight of size residual block vs ground block

# Ground-obs sources that directly annotate a ground marker (vs. vspan_proj,
# which is a downward extrapolation of a vertical span). Only direct markers
# pin camera roll. Markers along a single transect are collinear on the ground,
# and a rotation about the optical axis maps that line onto itself up to a
# compensating change in pitch and focal length, so roll is unidentifiable from
# one transect however many markers it holds. Two transects at different
# azimuths break the degeneracy; fit() therefore holds roll at zero below that.
DIRECT_GROUND_SOURCES = ("wire_point", "f2g_end")


class GroundPlaneFit:
    def __init__(self, params, cx, cy, ok, d_min=None, d_max=None, corrections=None):
        self.params, self.cx, self.cy, self.ok = params, cx, cy, ok
        self.d_min, self.d_max = d_min, d_max
        self.corrections = corrections or {}

    @property
    def frame(self):
        """(width, height) of the photo this calibration was fitted on."""
        return int(round(2 * self.cx)), int(round(2 * self.cy))

    @classmethod
    def fit(cls, photo):
        cx, cy = photo.image_w / 2.0, photo.image_h / 2.0
        g = photo.ground
        if len(g) < MIN_GROUND_OBS or len({o.dist for o in g}) < MIN_DISTINCT_DISTS:
            return cls(None, cx, cy, ok=False)
        gu = np.array([o.u for o in g]); gv = np.array([o.v for o in g])
        gd = np.array([o.dist for o in g]); gw = np.array([o.weight for o in g])
        s = photo.size
        sd = np.array([o.dist for o in s]); sl = np.array([o.px_len for o in s])
        scm = np.array([o.cm_len for o in s]); sw = np.array([o.weight for o in s])
        svert = np.array([o.vertical for o in s])

        # roll is unidentifiable from a single transect (see DIRECT_GROUND_SOURCES)
        n_direct = len({o.transect for o in g if o.source in DIRECT_GROUND_SOURCES})
        fix_roll = n_direct <= 1

        def residuals(q):
            p = PlaneParams(q[0], q[1], q[2], 0.0 if fix_roll else q[3])
            pred_d = pixel_to_distance(gu, gv, p, cx, cy)
            rg = np.log(np.where(np.isfinite(pred_d) & (pred_d > 0), pred_d, 1e6) / gd) * gw
            if len(s):
                rs = np.log(flag_span_px(sd, scm, p, svert) / sl) * sw * SIZE_TERM_WEIGHT
                return np.concatenate([rg, rs])
            return rg

        n = 3 if fix_roll else 4                 # (f, h, pitch[, roll])
        res = least_squares(residuals, x0=[3000.0, 1.0, 0.1, 0.0][:n],
                            bounds=([800.0, 0.2, -0.5, -0.4][:n], [8000.0, 4.0, 0.7, 0.4][:n]),
                            loss="soft_l1", f_scale=0.1)
        p = PlaneParams(res.x[0], res.x[1], res.x[2], 0.0 if fix_roll else res.x[3])
        m = cls(p, cx, cy, ok=True, d_min=float(gd.min()), d_max=float(gd.max()))
        m.corrections = m._build_corrections(photo)
        return m

    def _azimuth(self, u, v):
        xc = (u - self.cx) / self.params.f
        yc = (v - self.cy) / self.params.f
        xw, _, zw = _cam_to_world(xc, yc, 1.0, self.params)
        return np.arctan2(xw, zw)

    def _build_corrections(self, photo):
        corrections = {}
        for name in "LCR":
            obs = [o for o in photo.ground if o.transect == name]
            per_flag = {}
            for o in obs:
                per_flag.setdefault(o.dist, []).append(o)
            knots = []
            azs = []
            for d, flag_obs in per_flag.items():
                u = np.array([o.u for o in flag_obs]); v = np.array([o.v for o in flag_obs])
                preds = pixel_to_distance(u, v, self.params, self.cx, self.cy)
                preds = preds[np.isfinite(preds) & (preds > 0)]
                if not preds.size:
                    continue
                pf = float(np.median(preds))
                knots.append((pf, math.log(pf / d)))
                azs.extend(self._azimuth(u, v))
            if len(knots) >= 2:
                knots.sort()
                corrections[name] = ([k[0] for k in knots], [k[1] for k in knots],
                                     float(np.mean(azs)))
        return corrections

    def _correction(self, az, d):
        anchors = []
        for xs, ys, t_az in self.corrections.values():
            c = float(np.interp(d, xs, ys)) if xs[0] <= d <= xs[-1] else 0.0
            anchors.append((t_az, c))
        if not anchors:
            return 0.0
        return float(np.interp(az, *zip(*sorted(anchors))))

    def predict(self, u, v):
        if not self.ok:
            return None, "insufficient_data"
        d = float(pixel_to_distance(np.array([u]), np.array([v]), self.params,
                                    self.cx, self.cy)[0])
        if not np.isfinite(d) or d <= 0:
            return None, "above_horizon"
        d = d * math.exp(-self._correction(self._azimuth(u, v), d))
        if d < self.d_min:
            return d, "below_range"
        if d > self.d_max:
            return d, "beyond_range"
        return d, "ok"

    def to_dict(self):
        return {"ok": self.ok, "cx": self.cx, "cy": self.cy,
                "d_min": self.d_min, "d_max": self.d_max,
                "params": None if not self.ok else vars(self.params),
                "corrections": {k: list(map(list, v[:2])) + [v[2]]
                                for k, v in self.corrections.items()}}

    @classmethod
    def from_dict(cls, d):
        p = PlaneParams(**d["params"]) if d["params"] else None
        corr = {k: (v[0], v[1], v[2]) for k, v in d["corrections"].items()}
        return cls(p, d["cx"], d["cy"], d["ok"], d["d_min"], d["d_max"], corr)


def load_calibration(site, stem, root=CALIB_DIR):
    """Fitted calibration for one flag photo, or None if it is missing or failed."""
    p = root / site / f"{stem}.json"
    if not p.exists():
        return None
    m = GroundPlaneFit.from_dict(json.loads(p.read_text()))
    return m if m.ok else None
