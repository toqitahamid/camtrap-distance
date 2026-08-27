#!/usr/bin/env python3
"""Score a checkpoint on the held-out test cameras.

Each test flag photo is scored with its sibling flag photo as the deployment
reference (RoMa-aligned D_R from --roma-dir). Predictions are read at full
1920x1080 resolution with a 5x5 median at each surveyed flag pixel. Reports
MAE and p90 of the q50 channel, and coverage, Winkler score and mean width
(MPIW) of the [q05, q95] band.

Usage:
  PYTHONPATH=. python src/network/evaluate.py --ckpt outputs/ckpt/best --roma-dir outputs/prompts_roma
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from src.calibration.camera import load_cameras
from src.calibration.data import flag_pixels
from src.network.build_targets import TEST_SITES
from src.network.common import (SIZE, load_distance_map, load_imagenet_tensor, interval_metrics,
                                network_input, read_prediction)
from src.network.model import load_checkpoint


@torch.no_grad()
def score_cameras(model, sites, S, roma_dir):
    """Returns per-flag rows: dict(site,dist,q05,q50,q95). Deployment ref = sibling
    other-season photo + its D_R (or RoMa-warped prompt keyed by target stem)."""
    rows = []
    for site, cam in load_cameras(sites).items():
        for tgt in cam.photos:
            refs = cam.references_for(tgt)
            if not refs or not cam.distance_map(refs[0]).exists():
                continue
            ref = refs[0]
            img_t = load_imagenet_tensor(cam.photo_path(tgt), S)
            img_r = load_imagenet_tensor(cam.photo_path(ref), S)
            if roma_dir:
                wnpz = cam.aligned_prompt(tgt, roma_dir)
                if not wnpz.exists():
                    continue
                D_R = load_distance_map(wnpz, S)
            else:
                D_R = load_distance_map(cam.distance_map(ref), S)
            x = network_input(img_t, img_r, D_R)[None].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(pixel_values=x).predicted_depth               # (1,3,S,S)
            pred = read_prediction(pred, (tgt.image_h, tgt.image_w))       # (3,H,W)
            for (dist, transect), (u, v) in sorted(flag_pixels(tgt).items()):
                iu, iv = int(round(u)), int(round(v))
                patch = pred[:, max(0, iv - 2):iv + 3, max(0, iu - 2):iu + 3]
                with warnings.catch_warnings():        # all-NaN patch (sky) -> NaN, silently
                    warnings.simplefilter("ignore", RuntimeWarning)
                    vals = np.nanmedian(patch.reshape(3, -1), axis=1)
                if not np.all(np.isfinite(vals)):
                    print(f"NAN {site}/{Path(tgt.image).stem} {transect}{dist}")
                    continue
                rows.append(dict(site=site, dist=float(dist),
                                 img=Path(tgt.image).stem, tr=f"{transect}{dist}",
                                 q05=float(vals[0]), q50=float(vals[1]), q95=float(vals[2])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/ckpt/best")
    ap.add_argument("--roma-dir", default="outputs/prompts_roma",
                    help="dir of <site>__<target_stem>.npz RoMa-warped D_R (paper condition)")
    ap.add_argument("--csv", default=None, help="write test per-flag rows here")
    a = ap.parse_args()
    print(f"# ckpt {a.ckpt} roma-dir {a.roma_dir} test_sites {sorted(TEST_SITES)}", flush=True)
    test = score_cameras(load_checkpoint(a.ckpt, "cuda"), TEST_SITES, SIZE, a.roma_dir)
    tq05, tq50, tq95, ty = (np.array([r[k] for r in test]) for k in ("q05", "q50", "q95", "dist"))

    # ---- (a) POINT distance: MAE / p90 of q50 on TEST ----
    perr = np.abs(tq50 - ty)
    print(f"\n[POINT] TEST q50  median|err| {np.median(perr):.3f} m  "
          f"MAE {perr.mean():.3f}  p90 {np.percentile(perr, 90):.3f}  (n={perr.size})")
    for lo, hi in ((0, 5), (5, 10), (10, 16)):
        band = perr[(ty >= lo) & (ty < hi)]
        if band.size:
            print(f"          {lo:>2}-{hi}m median {np.median(band):.3f} (n={band.size})")

    # ---- (b) INTERVAL: [q05,q95] straight from the network head ----
    c0, w0, m0 = interval_metrics(tq05, tq95, ty)
    print(f"\n[INTERVAL] direct [q05,q95] : cov {c0:.3f}  Winkler {w0:.3f}  "
          f"MPIW {m0:.2f} m  (target cov 0.90)")

    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["site", "dist", "q05", "q50", "q95"],
                                extrasaction="ignore")
            w.writeheader(); w.writerows(test)
        print(f"# wrote {len(test)} test rows -> {a.csv}")


if __name__ == "__main__":
    main()
