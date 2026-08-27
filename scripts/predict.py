#!/usr/bin/env python3
"""Predict a metric distance map with a 90% interval for one camera-trap photo.

Inputs are the trained checkpoint, the new photo, and the camera's stored
reference: its flag photo plus that photo's calibration JSON (written by
scripts/calibrate.py).

By default the reference is RoMa-aligned onto the new photo before it is fed to
the network (the paper's aligned-reference path, built the same way as the
benchmark prompts), and the registration check reports when the alignment is
too weak to trust. Pass --no-align to feed the unaligned reference instead:
the paper's unaligned-reference ablation, markedly less accurate.

Output is an .npz holding q05/q50/q95 distance maps in meters at the photo's
resolution, plus the registration signals, and optionally a PNG of the median
map. To score many photos from Python, load the model once and call
src.network.predict.predict per photo.

Usage:
  PYTHONPATH=. python scripts/predict.py \
      --ckpt outputs/ckpt/best \
      --target path/to/new_photo.jpg \
      --ref data/flaglabel-dataset/MAS_CAM04/IMG_0001.JPG \
      --calibration data/calibrations/MAS_CAM04/IMG_0001.json \
      --out prediction.npz --png prediction.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.alignment.registration import describe_registration
from src.calibration.ground_plane import GroundPlaneFit
from src.network.model import load_checkpoint
from src.network.predict import predict


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="trained checkpoint dir (e.g. outputs/ckpt/best)")
    ap.add_argument("--target", required=True, help="new flag-free photo to score")
    ap.add_argument("--ref", required=True, help="the camera's stored reference flag photo")
    ap.add_argument("--calibration", required=True,
                    help="the reference photo's calibration JSON (data/calibrations/<camera>/<photo>.json)")
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="feed the raw D_R instead of RoMa-aligning it "
                         "(the paper's unaligned-reference ablation)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero instead of writing output when the registration check abstains")
    ap.add_argument("--out", default="prediction.npz", help="output .npz path")
    ap.add_argument("--png", default=None, help="optional PNG of the median map")
    a = ap.parse_args()

    calibration = GroundPlaneFit.from_dict(json.loads(Path(a.calibration).read_text()))
    if not calibration.ok:
        sys.exit(f"{a.calibration}: the calibration fit failed; re-survey this camera")
    p = predict(load_checkpoint(a.ckpt), a.target, a.ref, calibration, align=a.align)

    if p.registration is not None:
        print(describe_registration(p.registration))
        if p.abstain:
            print("  the stored reference no longer registers to this photo; "
                  "treat the prediction as unreliable and re-survey the camera.")
            if a.strict:
                sys.exit(2)
    else:
        print("unaligned-reference mode (--no-align): unaligned D_R, no registration check. "
              "This is the paper's ablation, not the shipped method.")

    extra = {f"gate_{k}": np.array(v) for k, v in (p.registration or {}).items()}
    np.savez_compressed(a.out, q05=p.q05, q50=p.q50, q95=p.q95, aligned=np.array(a.align), **extra)
    print(f"wrote {a.out}  ({p.q50.shape[1]}x{p.q50.shape[0]}, meters)")
    print(f"scene median distance {np.nanmedian(p.q50):.2f} m, "
          f"median interval width {np.nanmedian(p.q95 - p.q05):.2f} m")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(p.q50, cmap="viridis", vmin=0, vmax=min(20, float(np.nanmax(p.q50))))
        fig.colorbar(im, ax=ax, label="horizontal ground distance (m)")
        ax.set_axis_off()
        fig.savefig(a.png, bbox_inches="tight", dpi=150)
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
