import numpy as np
import pytest
from src.calibration.ground_plane import PlaneParams, pixel_to_distance, world_to_pixel, flag_span_px, FLAG_BODY_CENTER_HEIGHT_M

P = PlaneParams(f=3000.0, h=1.2, pitch=0.12, roll=0.02)
CX, CY = 960.0, 540.0

def test_roundtrip_ground_point():
    for d, az in [(2.0, -0.15), (7.0, 0.0), (15.0, 0.2)]:
        x, y, z = d * np.sin(az), P.h, d * np.cos(az)  # point ON the ground plane
        u, v = world_to_pixel(x, y, z, P, CX, CY)
        got = pixel_to_distance(np.array([u]), np.array([v]), P, CX, CY)[0]
        assert got == pytest.approx(d, rel=1e-9)

def test_above_horizon_is_nan():
    # straight up relative to optical axis: v far above the horizon row
    got = pixel_to_distance(np.array([960.0]), np.array([0.0]), P, CX, CY)[0]
    assert np.isnan(got)

def test_monotone_in_v():
    vs = np.linspace(600.0, 1050.0, 50)
    ds = pixel_to_distance(np.full(50, 960.0), vs, P, CX, CY)
    assert np.all(np.diff(ds) < 0)  # lower in image = closer

def test_predicted_px_shrinks_with_distance():
    near = flag_span_px(2.0, 6.35, P, vertical=True)
    far = flag_span_px(15.0, 6.35, P, vertical=True)
    assert near > far
    # far away, vertical ~ f * L / D (cos factor -> 1)
    assert far == pytest.approx(3000.0 * 0.0635 / 15.0, rel=0.02)

def test_predicted_px_vertical_foreshortening():
    # at close range the vertical span foreshortens vs the naive f*L/r
    d = 2.0
    r = np.hypot(d, P.h - FLAG_BODY_CENTER_HEIGHT_M)
    naive = 3000.0 * 0.0635 / r
    assert flag_span_px(d, 6.35, P, vertical=True) < naive


from src.calibration.ground_plane import GroundPlaneFit
from src.calibration.data import PhotoData
from tests.synth import projective_photo

def test_fit_recovers_parameters():
    ph, true_p = projective_photo(noise_px=1.0)
    m = GroundPlaneFit.fit(ph)
    assert m.ok
    assert m.params.f == pytest.approx(true_p.f, rel=0.05)
    assert m.params.h == pytest.approx(true_p.h, rel=0.05)
    assert m.params.pitch == pytest.approx(true_p.pitch, abs=0.02)

def test_fit_predicts_held_out_geometry():
    ph, true_p = projective_photo(noise_px=1.0)
    m = GroundPlaneFit.fit(ph)
    # a fresh ground point not in the observations: 9.5 m, azimuth 5 deg
    x, z = 9.5 * np.sin(np.radians(5)), 9.5 * np.cos(np.radians(5))
    u, v = world_to_pixel(x, true_p.h, z, true_p, 960.0, 540.0)
    d, status = m.predict(u, v)
    assert status == "ok" and d == pytest.approx(9.5, rel=0.05)

def test_fit_guard():
    ph = PhotoData("X", "X.JPG", 1920, 1080)  # zero observations
    m = GroundPlaneFit.fit(ph)
    assert not m.ok
    assert m.predict(960.0, 800.0) == (None, "insufficient_data")

def test_extrapolation_tagged():
    ph, true_p = projective_photo(dists=range(2, 11))  # calibrated only to 10 m
    m = GroundPlaneFit.fit(ph)
    # pixel of a 20 m ground point on the C azimuth
    u, v = world_to_pixel(0.0, true_p.h, 20.0, true_p, 960.0, 540.0)
    d, status = m.predict(u, v)
    assert status == "beyond_range" and d == pytest.approx(20.0, rel=0.1)

def test_above_horizon():
    ph, _ = projective_photo()
    m = GroundPlaneFit.fit(ph)
    assert m.predict(960.0, 0.0) == (None, "above_horizon")

def test_serialization_roundtrip():
    ph, _ = projective_photo(noise_px=1.0)
    m = GroundPlaneFit.fit(ph)
    m2 = GroundPlaneFit.from_dict(m.to_dict())
    assert m2.predict(1000.0, 900.0) == m.predict(1000.0, 900.0)
