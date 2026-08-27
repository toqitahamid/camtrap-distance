"""Load FlagLabel annotation JSONs into uniform calibration observations."""
import json
import math

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

PHOTOS_DIR = Path("data/flaglabel-dataset")   # fetched by scripts/fetch_data.py
CALIB_DIR = Path("data/calibrations")        # written by scripts/calibrate.py
TARGETS_DIR = Path("outputs/targets_d")      # written by src/network/build_targets.py
PROMPTS_DIR = Path("outputs/prompts_roma")   # written by src/alignment/build_prompts.py

LEAN_MAX_DEG = 45.0
W_GROUND_DIRECT = 1.0
W_GROUND_PROJECTED = 0.5
W_SIZE_BODY = 1.0
W_SIZE_FLAG_TO_GROUND = 0.3  # flag-to-ground length is a fleet average (burial depth varies)


@dataclass
class GroundObs:
    u: float
    v: float
    dist: float
    transect: str
    source: str
    weight: float


@dataclass
class SizeObs:
    u: float
    v: float
    px_len: float
    cm_len: float
    dist: float
    transect: str
    source: str
    weight: float
    vertical: bool


@dataclass
class PhotoData:
    site: str
    image: str
    image_w: int
    image_h: int
    ground: list = field(default_factory=list)
    size: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _span_length_px(s):
    return math.hypot(s["u2"] - s["u1"], s["v2"] - s["v1"])


def _span_midpoint(s):
    return (s["u1"] + s["u2"]) / 2, (s["v1"] + s["v2"]) / 2


def load_photo(path):
    d = json.loads(Path(path).read_text())
    ref = d.get("reference_dimensions_cm") or {}
    body_h = ref.get("flag_body_h", 6.35)
    body_w = ref.get("flag_body_w", 8.89)
    wire_ag = ref.get("wire_above_ground", 49.53)
    bottom_cm = wire_ag - body_h  # height of flag-body bottom above ground

    ph = PhotoData(d["site"], d["image"], d["image_w"], d["image_h"])

    for p in d.get("wire_ground_points") or []:
        ph.ground.append(GroundObs(p["u"], p["v"], p["distance"], p["transect"],
                                   "wire_point", W_GROUND_DIRECT))

    for s in d.get("flag_to_ground_spans") or []:
        # schema-ordered: endpoint 1 = flag top, endpoint 2 = ground
        ph.ground.append(GroundObs(s["u2"], s["v2"], s["distance"], s["transect"],
                                   "f2g_end", W_GROUND_DIRECT))
        L = _span_length_px(s)
        if L > 0:
            mu, mv = _span_midpoint(s)
            ph.size.append(SizeObs(mu, mv, L, wire_ag, s["distance"], s["transect"],
                                   "f2g_len", W_SIZE_FLAG_TO_GROUND, True))

    for s in d.get("flag_horizontal_spans") or []:
        L = _span_length_px(s)
        if L > 0:
            mu, mv = _span_midpoint(s)
            ph.size.append(SizeObs(mu, mv, L, body_w, s["distance"], s["transect"],
                                   "hspan", W_SIZE_BODY, False))

    for s in d.get("flag_vertical_spans") or []:
        L = _span_length_px(s)
        if L <= 0:
            continue
        mu, mv = _span_midpoint(s)
        ph.size.append(SizeObs(mu, mv, L, body_h, s["distance"], s["transect"],
                               "vspan", W_SIZE_BODY, True))
        # ground projection: continue the span axis (the wire direction) downward
        # from the bottom endpoint; endpoint order is not guaranteed, bottom = larger v
        (ut, vt), (ub, vb) = sorted([(s["u1"], s["v1"]), (s["u2"], s["v2"])],
                                    key=lambda p: p[1])
        dx, dy = (ub - ut) / L, (vb - vt) / L  # unit vector, top -> bottom
        lean_deg = math.degrees(math.atan2(abs(dx), dy))
        if lean_deg > LEAN_MAX_DEG:
            ph.skipped.append({"source": "vspan", "distance": s["distance"],
                               "transect": s["transect"], "lean_deg": lean_deg})
            continue
        drop_px = bottom_cm * L / body_h  # local scale: L px per body_h cm
        ph.ground.append(GroundObs(ub + dx * drop_px, vb + dy * drop_px,
                                   s["distance"], s["transect"],
                                   "vspan_proj", W_GROUND_PROJECTED))
    return ph


def load_dataset(root=PHOTOS_DIR):
    return [load_photo(p) for p in sorted(Path(root).glob("*/*.json"))]


def markers(photo):
    """Ground observations grouped per surveyed flag: (dist, transect) -> [GroundObs]."""
    per_flag = {}
    for g in photo.ground:
        per_flag.setdefault((g.dist, g.transect), []).append(g)
    return per_flag


def flag_pixels(photo):
    """Mean annotated pixel of each flag: (dist, transect) -> (u, v)."""
    return {k: (float(np.mean([g.u for g in v])), float(np.mean([g.v for g in v])))
            for k, v in markers(photo).items()}
