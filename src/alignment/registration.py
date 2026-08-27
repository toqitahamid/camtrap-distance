"""Deployment-time registration: align a camera's stored reference onto a new photo.

A camera is surveyed once; its flag photo and calibrated distance map D_R are
stored. The housing drifts by a few degrees over months, so before each new
photo is scored RoMa matches it against the stored flag photo, a homography is
fitted to the matches, and D_R is warped into the new photo's frame.
`build_prompts.py` does the same for flag-photo pairs inside the benchmark.

A registration check decides whether to trust the result. Two signals come
from the match itself:

  inliers  RANSAC inliers supporting the homography. Too few means the scene
           changed too much to register (snow, leaf-off, a fallen branch).
  ratio    mean warp displacement over a ground-region probe grid, divided by
           the median displacement of the matches themselves. A homography that
           moves the frame much further than its matches moved is extrapolating
           into parts of the image it has no evidence for.

Either signal tripping means abstain and flag the camera for a fresh survey.
"""
import cv2
import numpy as np
from PIL import Image

from src.alignment.matchers import CROPS_DIR, banner_cropped, fit_homography, match_roma

# Thresholds were fitted on LoFTR-registered flag-photo pairs to catch every
# registration failure with a >10 m error. Under RoMa, the deployed matcher,
# neither fires on any benchmark pair, so the check is a safeguard for
# conditions the survey never samples (snow, a fouled lens, a moved camera)
# rather than a routinely active filter.
REGISTRATION_MIN_INLIERS = 15
REGISTRATION_MAX_WARP_RATIO = 2.33

# Ground-region probe grid the warp ratio is measured over: the lower-middle of
# a 1920x1080 frame, where the ground plane actually appears. Scaled to the
# target's own size for other resolutions.
_PROBE_X = np.linspace(150, 1770, 8) / 1920.0
_PROBE_Y = np.linspace(400, 990, 6) / 1080.0


def _probe_grid(width, height):
    return np.array([(x * width, y * height) for y in _PROBE_Y for x in _PROBE_X])


def registration_verdict(n_inliers, warp_ratio):
    """Should this registration be trusted? Returns (abstain, reason).

    Kept separate from the matching so the decision can be checked without a
    GPU. A NaN ratio (no usable matches to compare against) is not by itself a
    reason to abstain; the inlier count catches that case.
    """
    reasons = []
    if n_inliers < REGISTRATION_MIN_INLIERS:
        reasons.append(f"only {n_inliers} match inliers (< {REGISTRATION_MIN_INLIERS})")
    if np.isfinite(warp_ratio) and warp_ratio > REGISTRATION_MAX_WARP_RATIO:
        reasons.append(f"warp/match ratio {warp_ratio:.2f} (> {REGISTRATION_MAX_WARP_RATIO})")
    return bool(reasons), "; ".join(reasons)


def _warp_points(homography, points):
    homogeneous = np.c_[points, np.ones(len(points))]
    warped = (homography @ homogeneous.T).T
    return warped[:, :2] / warped[:, 2:3]


def match_homography(from_photo, to_photo, matcher=match_roma, scratch_dir=CROPS_DIR):
    """Homography mapping pixels of `from_photo` into `to_photo`, from dense matches.

    `matcher(a, b)` follows match_roma's convention: it returns (points in b,
    points in a) on banner-cropped copies. Any callable with that shape can sit
    at this seam; tests use a synthetic one.

    Returns (homography or None, signals). `signals` carries the registration
    check's inputs and verdict; `abstain` is True whenever no homography could
    be fitted or the check tripped.
    """
    to_w, to_h = Image.open(to_photo).size
    src, dst = matcher(banner_cropped(to_photo, scratch_dir),      # src in from_photo,
                       banner_cropped(from_photo, scratch_dir))    # dst in to_photo
    n_matches = int(len(src))
    match_displacement = (float(np.median(np.linalg.norm(dst - src, axis=1)))
                          if n_matches else float("nan"))

    homography, n_inliers = fit_homography(src, dst)
    signals = dict(n_matches=n_matches, n_inliers=n_inliers,
                   match_displacement=match_displacement,
                   warp_magnitude=float("nan"), warp_ratio=float("nan"),
                   registered=False, abstain=True,
                   reason="no homography could be fitted")
    if homography is None:
        return None, signals

    probe = _probe_grid(to_w, to_h)
    warp_magnitude = float(np.mean(np.linalg.norm(_warp_points(homography, probe) - probe, axis=1)))
    warp_ratio = (warp_magnitude / match_displacement
                  if match_displacement and match_displacement > 0 else float("nan"))
    abstain, reason = registration_verdict(n_inliers, warp_ratio)
    signals.update(match_displacement=round(match_displacement, 3),
                   warp_magnitude=round(warp_magnitude, 3),
                   warp_ratio=round(warp_ratio, 3) if np.isfinite(warp_ratio) else float("nan"),
                   registered=True, abstain=abstain, reason=reason)
    return homography, signals


def register_reference(target_path, reference_path, reference_distance_map,
                       scratch_dir=CROPS_DIR, matcher=match_roma):
    """Warp a stored reference distance map into a new photo's frame.

    Args:
        target_path: the new flag-free photo to score.
        reference_path: the camera's stored flag photo.
        reference_distance_map: D_R for that flag photo, any resolution; it is
            resampled to the reference photo's native size before warping.
        scratch_dir: where banner-cropped copies are cached.
        matcher: see match_homography.

    Returns:
        (aligned_distance_map, signals). The map is float32 at the TARGET
        photo's native size, 0 where the warp maps outside the reference.
        When registration fails outright the map is all-zero and
        `signals["abstain"]` is True.
    """
    target_w, target_h = Image.open(target_path).size
    ref_w, ref_h = Image.open(reference_path).size

    plane = np.nan_to_num(np.asarray(reference_distance_map, dtype=np.float32), nan=0.0)
    if plane.shape != (ref_h, ref_w):
        plane = cv2.resize(plane, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

    homography, signals = match_homography(reference_path, target_path, matcher, scratch_dir)
    if homography is None:
        return np.zeros((target_h, target_w), np.float32), signals
    aligned = cv2.warpPerspective(plane, homography, (target_w, target_h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned.astype(np.float32), signals


def describe_registration(signals):
    """One-line human-readable summary of a registration."""
    if not signals["registered"]:
        return f"registration FAILED ({signals['reason']})"
    verdict = f"ABSTAIN — {signals['reason']}" if signals["abstain"] else "ok"
    return (f"registration {verdict} "
            f"[{signals['n_inliers']} inliers of {signals['n_matches']} matches, "
            f"warp/match ratio {signals['warp_ratio']}]")
