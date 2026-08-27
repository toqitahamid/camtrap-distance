"""Registration logic checked on a CPU through the matcher seam: a synthetic
matcher returns correspondences from a known homography, so the fitted warp,
the check's signals and the failure path are all exercised without RoMa."""
import numpy as np
import pytest
from PIL import Image

from src.alignment.registration import match_homography, register_reference

W, H = 320, 180
H_TRUE = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])   # pure shift


@pytest.fixture
def photos(tmp_path):
    paths = []
    for name in ("reference.png", "target.png"):
        p = tmp_path / name
        Image.new("RGB", (W, H), (90, 120, 60)).save(p)
        paths.append(str(p))
    return paths  # (reference, target)


def synthetic_matcher(n=300):
    """Mirrors match_roma(a, b): returns (points in b, points in a)."""
    def matcher(path_a, path_b):
        rng = np.random.default_rng(0)
        src = np.c_[rng.uniform(0, W, n), rng.uniform(0, H, n)]
        pts = np.c_[src, np.ones(n)] @ H_TRUE.T
        return src.astype(np.float32), (pts[:, :2] / pts[:, 2:3]).astype(np.float32)
    return matcher


def no_matches(path_a, path_b):
    return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)


def test_recovers_the_homography_and_passes_the_check(photos, tmp_path):
    ref, tgt = photos
    Hm, sig = match_homography(ref, tgt, matcher=synthetic_matcher(), scratch_dir=tmp_path)
    assert np.allclose(Hm / Hm[2, 2], H_TRUE, atol=1e-3)
    assert sig["registered"] and not sig["abstain"] and sig["reason"] == ""
    assert sig["n_inliers"] == sig["n_matches"] == 300
    assert sig["match_displacement"] == pytest.approx(5.0, abs=1e-3)   # |(4, -3)|
    assert sig["warp_ratio"] == pytest.approx(1.0, abs=1e-3)           # a shift moves everything equally


def test_aligned_map_is_the_reference_warped_into_the_target(photos, tmp_path):
    ref, tgt = photos
    plane = np.tile(np.linspace(2, 18, W, dtype=np.float32), (H, 1))
    aligned, sig = register_reference(tgt, ref, plane, scratch_dir=tmp_path,
                                      matcher=synthetic_matcher())
    assert aligned.shape == (H, W) and aligned.dtype == np.float32 and not sig["abstain"]
    # a +4 px shift in x: column 100 of the target reads what column 96 held in the reference
    assert aligned[50, 100] == pytest.approx(plane[50, 96], abs=1e-3)
    assert (aligned[:, :4] == 0).all()                                # nothing maps into the new border


def test_no_matches_means_abstain_and_an_all_zero_map(photos, tmp_path):
    ref, tgt = photos
    aligned, sig = register_reference(tgt, ref, np.full((H, W), 5.0, np.float32),
                                      scratch_dir=tmp_path, matcher=no_matches)
    assert sig["abstain"] and not sig["registered"] and "no homography" in sig["reason"]
    assert aligned.shape == (H, W) and not aligned.any()


def test_a_wrong_sized_reference_map_is_resampled_first(photos, tmp_path):
    ref, tgt = photos
    aligned, _ = register_reference(tgt, ref, np.full((45, 80), 7.0, np.float32),
                                    scratch_dir=tmp_path, matcher=synthetic_matcher())
    assert aligned.shape == (H, W) and aligned[50, 100] == pytest.approx(7.0)
