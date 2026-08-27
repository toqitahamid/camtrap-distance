#!/usr/bin/env python3
"""RoMa-aligned reference prompts for every two-photo camera.

For each ordered pair (A, B) of a camera's flag photos, A's calibrated distance
map D_R is warped by the RoMa homography into B's frame and written to
<out>/<site>__<stemB>.npz. Reads data/calibrations/ (scripts/calibrate.py), so
re-running the calibration means rebuilding the prompts of the cameras it
touched.

Usage (GPU):
  PYTHONPATH=. python src/alignment/build_prompts.py --out outputs/prompts_roma
  PYTHONPATH=. python src/alignment/build_prompts.py --sites MAS_CAM07 MAS_CAM08   # only these cameras
"""
import argparse
from pathlib import Path

import numpy as np

from src.calibration.camera import load_cameras
from src.calibration.data import PROMPTS_DIR
from src.calibration.ground_plane import dense_distance_map
from src.alignment.registration import match_homography


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=None,
                    help="camera dirs to regenerate; default = all 2-photo cameras")
    ap.add_argument("--out", default=str(PROMPTS_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for site, cam in load_cameras(set(args.sites) if args.sites else None).items():
        if len(cam.photos) != 2:
            continue
        ps = cam.photos
        for a, b in ((ps[0], ps[1]), (ps[1], ps[0])):     # a = reference, b = target
            ma = cam.calibration(a)
            if ma is None:
                continue
            # pixels of B (the target grid) -> A's calibrated frame
            Hm, sig = match_homography(cam.photo_path(b), cam.photo_path(a))
            if Hm is None:
                print(f"{site} {a.image}->{b.image}: NO HOMOGRAPHY ({sig['n_matches']} matches)", flush=True)
                continue
            ninl = sig["n_inliers"]
            d = dense_distance_map(ma.params, ma.cx, ma.cy, width=ma.frame[0], height=ma.frame[1],
                                   homography=Hm)
            np.savez_compressed(cam.aligned_prompt(b, out),
                                plane_d=d.astype(np.float16), inliers=ninl,
                                src_image=np.array(a.image))
            print(f"{site} {a.image}->{b.image}: inl {ninl}, "
                  f"valid px {np.isfinite(d).sum()}", flush=True)


if __name__ == "__main__":
    main()
