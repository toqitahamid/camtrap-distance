import json, math
from pathlib import Path
from src.calibration.data import load_photo, LEAN_MAX_DEG

def _write(tmp_path, payload):
    p = tmp_path / "SITE_CAM01" / "IMG_0001.json"
    p.parent.mkdir()
    p.write_text(json.dumps(payload))
    return p

BASE = {
    "schema_version": 2, "site": "SITE_CAM01", "image": "IMG_0001.JPG",
    "image_w": 1920, "image_h": 1080,
    "reference_dimensions_cm": {"flag_body_h": 6.35, "flag_body_w": 8.89,
        "wire_total": 53.34, "wire_above_ground": 49.53, "wire_buried": 3.81},
    "flag_vertical_spans": [], "flag_horizontal_spans": [],
    "flag_to_ground_spans": [], "wire_ground_points": [],
}

def test_direct_ground_sources(tmp_path):
    d = dict(BASE)
    d["wire_ground_points"] = [{"u": 100.0, "v": 800.0, "distance": 3, "transect": "L"}]
    d["flag_to_ground_spans"] = [
        {"u1": 500.0, "v1": 700.0, "u2": 505.0, "v2": 790.0, "distance": 5, "transect": "C"}]
    ph = load_photo(_write(tmp_path, d))
    assert ph.site == "SITE_CAM01" and ph.image_w == 1920
    assert len(ph.ground) == 2
    wp = next(g for g in ph.ground if g.source == "wire_point")
    assert (wp.u, wp.v, wp.dist, wp.transect, wp.weight) == (100.0, 800.0, 3, "L", 1.0)
    f2g = next(g for g in ph.ground if g.source == "f2g_end")
    assert (f2g.u, f2g.v) == (505.0, 790.0)  # endpoint 2 = ground
    # flag-to-ground length is a downweighted size obs vs 49.53 cm
    s = ph.size[0]
    assert s.cm_len == 49.53 and s.weight == 0.3 and s.vertical
    assert math.isclose(s.px_len, math.hypot(5.0, 90.0))

def test_vertical_span_projection_follows_lean(tmp_path):
    d = dict(BASE)
    # span leaning: top (200,700) -> bottom (210,760); length ~60.8 px for 6.35 cm
    d["flag_vertical_spans"] = [
        {"u1": 200.0, "v1": 700.0, "u2": 210.0, "v2": 760.0, "distance": 4, "transect": "C"}]
    ph = load_photo(_write(tmp_path, d))
    assert len(ph.size) == 1 and ph.size[0].cm_len == 6.35 and ph.size[0].weight == 1.0
    assert len(ph.ground) == 1
    g = ph.ground[0]
    L = math.hypot(10.0, 60.0)
    drop = 43.18 * L / 6.35          # px along the span axis
    assert math.isclose(g.u, 210.0 + (10.0 / L) * drop)
    assert math.isclose(g.v, 760.0 + (60.0 / L) * drop)
    assert g.weight == 0.5 and g.source == "vspan_proj"

def test_vertical_span_endpoint_order_irrelevant(tmp_path):
    d = dict(BASE)
    d["flag_vertical_spans"] = [   # endpoints swapped: point 1 is the BOTTOM
        {"u1": 210.0, "v1": 760.0, "u2": 200.0, "v2": 700.0, "distance": 4, "transect": "C"}]
    ph = load_photo(_write(tmp_path, d))
    g = ph.ground[0]
    assert g.v > 760.0  # still projects downward from the bottom endpoint

def test_extreme_lean_skipped(tmp_path):
    d = dict(BASE)
    # ~59° from vertical: dx=100, dy=60
    d["flag_vertical_spans"] = [
        {"u1": 200.0, "v1": 700.0, "u2": 300.0, "v2": 760.0, "distance": 7, "transect": "R"}]
    ph = load_photo(_write(tmp_path, d))
    assert len(ph.ground) == 0            # no ground projection
    assert len(ph.size) == 1              # size obs kept
    assert len(ph.skipped) == 1 and ph.skipped[0]["lean_deg"] > LEAN_MAX_DEG

def test_empty_arrays_and_missing_keys(tmp_path):
    d = dict(BASE)
    del d["wire_ground_points"]
    ph = load_photo(_write(tmp_path, d))
    assert ph.ground == [] and ph.size == [] and ph.skipped == []
