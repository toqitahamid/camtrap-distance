from src.calibration.qc import monotonicity, leave_one_flag_out, cross_photo
from src.calibration.ground_plane import GroundPlaneFit
from tests.synth import projective_photo

def test_monotonicity_clean():
    ph, _ = projective_photo()
    assert monotonicity(ph) == []

def test_monotonicity_catches_swapped_labels():
    ph, _ = projective_photo()
    # swap the distance tags of the 5 m and 6 m flags on C
    for g in ph.ground:
        if g.transect == "C" and g.dist == 5.0:
            g.dist = 6.0
        elif g.transect == "C" and g.dist == 6.0:
            g.dist = 5.0
    v = monotonicity(ph)
    assert any(x["transect"] == "C" and x["kind"] == "ground_v" for x in v)

def test_leave_one_flag_out_small_errors_on_synthetic():
    ph, _ = projective_photo(noise_px=1.0)
    rows = leave_one_flag_out(ph)
    assert len(rows) == 42
    errs_b = [abs(r["err_b"]) for r in rows if r["err_b"] is not None]
    assert len(errs_b) > 35
    errs_b.sort()
    assert errs_b[len(errs_b) // 2] < 0.5      # median LOO error under 0.5 m

def test_cross_photo():
    ph1, _ = projective_photo(noise_px=1.0, seed=1)
    ph2, _ = projective_photo(noise_px=1.0, seed=2, pitch=0.15)  # drifted aim, same camera
    m1, m2 = GroundPlaneFit.fit(ph1), GroundPlaneFit.fit(ph2)
    rows = cross_photo({"SYN": [m1, m2]})
    assert len(rows) == 1 and rows[0]["f_spread_pct"] < 15.0
