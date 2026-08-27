"""Self-check for the roll-identifiability prior in GroundPlaneFit.fit.

Rule (see DIRECT_GROUND_SOURCES in src/calibration/ground_plane.py): a photo
with <=1 distinct DIRECT-marker transect (source wire_point / f2g_end) cannot identify camera
roll, so roll is hard-fixed to 0 and only f/h/pitch are fitted; a photo with
>=2 direct transects fits all 4 params (roll free) exactly as before.

Run standalone:
    PYTHONPATH=. python tests/test_roll_identifiability.py
Or via pytest.
"""
import math

from src.calibration.data import GroundObs, PhotoData
from src.calibration.ground_plane import GroundPlaneFit, PlaneParams, world_to_pixel

CX, CY = 960.0, 540.0
H_TRUE = 2.0
F_TRUE = 3000.0
PITCH_TRUE = 0.2


def _obs(x, z, p_true, transect):
    """Project a ground world point (y = h) to a direct GroundObs."""
    u, v = world_to_pixel(x, H_TRUE, z, p_true, CX, CY)
    dist = math.hypot(x, z)
    return GroundObs(u, v, dist, transect, "wire_point", 1.0)


def _photo(ground):
    ph = PhotoData("SYNTH", "img.jpg", 1920, 1080)
    ph.ground = ground
    return ph


def test_three_transects_fits_roll_freely():
    """3 direct transects -> roll is fitted (recovers the injected nonzero roll)."""
    p_true = PlaneParams(F_TRUE, H_TRUE, PITCH_TRUE, 0.12)
    ground = []
    for x, transect in [(-2.0, "L"), (0.0, "C"), (2.0, "R")]:
        for z in (5.0, 8.0, 11.0):
            ground.append(_obs(x, z, p_true, transect))
    m = GroundPlaneFit.fit(_photo(ground))
    assert m.ok, "3-transect synthetic photo should fit"
    assert abs(m.params.roll - 0.12) < 1e-3, (
        f"roll should recover ~0.12 freely, got {m.params.roll}")


def test_one_transect_fixes_roll_to_zero():
    """<=1 direct transect -> roll hard-fixed to exactly 0.0, f/h/pitch still fit."""
    p_true = PlaneParams(F_TRUE, H_TRUE, PITCH_TRUE, 0.0)
    ground = [_obs(0.0, z, p_true, "C") for z in (4.0, 5.0, 6.0, 8.0, 10.0, 12.0)]
    m = GroundPlaneFit.fit(_photo(ground))
    assert m.ok, "1-transect synthetic photo should still fit (3 free params)"
    assert m.params.roll == 0.0, f"roll must be exactly 0.0, got {m.params.roll}"
    assert abs(m.params.h - H_TRUE) < 1e-2, f"height off: {m.params.h}"
    assert abs(m.params.pitch - PITCH_TRUE) < 1e-2, f"pitch off: {m.params.pitch}"


if __name__ == "__main__":
    test_three_transects_fits_roll_freely()
    test_one_transect_fixes_roll_to_zero()
    print("PASS: roll fitted freely with 3 transects; hard-fixed to 0 with 1 transect")
