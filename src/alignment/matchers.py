"""RoMa dense matching between a camera's reference and target photos.

Used by build_prompts.py to warp the reference distance map D_R into each
target frame. The camera's info banner (rows below BANNER_Y) is identical
across photos and would produce false matches, so photos are matched on
banner-cropped copies.
"""
from pathlib import Path

import cv2
from PIL import Image

BANNER_Y = 995  # camera info strip below this row is identical across photos
CROPS_DIR = "outputs/banner_crops"   # cache of banner-cropped copies
RANSAC_REPROJ_PX = 3.0
MIN_MATCHES = 4                      # a homography needs four correspondences

MATCH_SEED = 0                       # see match_roma


def banner_cropped(photo_path, scratch_dir=CROPS_DIR):
    """Copy of a photo with the camera's info banner removed, cached in scratch_dir.

    The banner (date, temperature, camera name) is burned into the bottom rows
    and is nearly identical across photos, so it produces confident false
    matches. Cropping only removes bottom rows, so pixel coordinates in the kept
    region are unchanged and the homography stays valid in the full frame.
    """
    photo_path, scratch_dir = Path(photo_path), Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out = scratch_dir / f"{photo_path.parent.name}__{photo_path.stem}_nobanner.png"
    if not out.exists():
        im = Image.open(photo_path).convert("RGB")
        im.crop((0, 0, im.width, min(BANNER_Y, im.height))).save(out)
    return str(out)


_roma = None


def _get_roma():
    global _roma
    if _roma is None:
        import torch
        from romatch import roma_outdoor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # custom local_corr CUDA kernel isn't built here; pure-torch fallback
        _roma = (roma_outdoor(device=dev, use_custom_corr=False), dev)
    return _roma


def match_roma(path_a, path_b):
    """RoMa symmetric warp -> sampled A/B pixel matches. Returns src(B), dst(A).

    `model.sample` draws matches at random from the certainty map, so the RNG is
    seeded per call (and its state restored afterwards) to keep prompt building
    reproducible without disturbing a caller's own RNG stream.
    """
    import torch
    model, dev = _get_roma()
    W_A, H_A = Image.open(path_a).size
    W_B, H_B = Image.open(path_b).size
    warp, cert = model.match(path_a, path_b, device=dev)

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(MATCH_SEED)
        matches, cert = model.sample(warp, cert)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    kA, kB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    src = kB.cpu().numpy()  # in B
    dst = kA.cpu().numpy()  # in A
    return src, dst


def fit_homography(src, dst):
    """Robust (USAC-MAGSAC) homography src -> dst. Returns (H or None, n_inliers)."""
    if len(src) < MIN_MATCHES:
        return None, 0
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, RANSAC_REPROJ_PX)
    return H, (int(mask.sum()) if mask is not None else 0)
