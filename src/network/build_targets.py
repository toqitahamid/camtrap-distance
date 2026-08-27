#!/usr/bin/env python3
"""D_R targets: per-photo plane-distance map + flag anchors, built from the
fitted camera parameters in data/calibrations/, as written by
scripts/calibrate.py. Also the single home of the camera split (VAL_SITES,
TEST_SITES, split_of), which training, evaluation and the tests all read.

Usage: PYTHONPATH=. python src/network/build_targets.py
"""
import numpy as np

from src.calibration.camera import load_cameras
from src.calibration.data import TARGETS_DIR, flag_pixels
from src.calibration.ground_plane import dense_distance_map

VAL_SITES = {"MAS_CAM01", "MAS_CAM03", "MAS_CAM25", "TON_CAM06",
             "TON_CAM14", "TON_CAM16", "TON_CAM26"}
TEST_SITES = {"MAS_CAM04", "MAS_CAM12", "MAS_CAM13", "MAS_CAM17", "MAS_CAM21",
              "MAS_CAM24", "TON_CAM04", "TON_CAM11", "TON_CAM13", "TON_CAM19",
              "TON_CAM24", "TON_CAM32"}


def split_of(site):
    return "test" if site in TEST_SITES else "val" if site in VAL_SITES else "train"


def main():
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for cam in load_cameras().values():
        for photo in cam.photos:
            m = cam.calibration(photo)
            if m is None:
                continue
            d = dense_distance_map(m.params, m.cx, m.cy, width=m.frame[0], height=m.frame[1])
            fuv, fd = [], []
            for (dist, transect), (u, v) in sorted(flag_pixels(photo).items()):
                fuv.append((u, v))
                fd.append(float(dist))
            np.savez_compressed(cam.distance_map(photo),
                                plane_d=d.astype(np.float16),
                                flag_uv=np.array(fuv, dtype=np.float32),
                                flag_d=np.array(fd, dtype=np.float32))
            n += 1
    print("done:", n, "files")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
