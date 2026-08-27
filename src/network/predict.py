"""Deployment: a metric distance map with a 90% interval for one camera-trap photo.

A camera is surveyed once. Its flag photo and that photo's fitted calibration
(scripts/calibrate.py) are the stored reference; every later photo from the
camera is scored against them. Load the model once with `load_checkpoint` and
call `predict` per photo.

The reference distance map is carried into the new photo exactly as the
benchmark prompts are built (src/alignment/build_prompts.py): RoMa fits the
homography from the new photo to the flag photo, and the calibration is
evaluated on the new photo's pixel grid mapped through it.
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from src.alignment.matchers import CROPS_DIR
from src.alignment.registration import match_homography
from src.calibration.ground_plane import dense_distance_map
from src.network.common import (SIZE, distance_map_tensor, load_imagenet_tensor, network_input,
                                read_prediction)


@dataclass
class Prediction:
    q05: np.ndarray            # (H, W) metres, at the target photo's resolution
    q50: np.ndarray
    q95: np.ndarray
    registration: dict | None  # registration signals; None when align=False

    @property
    def abstain(self):
        """True when the stored reference no longer registers to this photo."""
        return bool(self.registration and self.registration["abstain"])


def reference_prompt(target, reference, calibration, align=True, scratch_dir=CROPS_DIR):
    """The D_R prompt channel at model resolution: the reference calibration
    evaluated on the target's pixel grid, mapped through the RoMa homography
    (align=True, the paper's aligned-reference path) or unmapped (align=False,
    the unaligned-reference ablation). Returns (prompt, signals)."""
    homography, signals = None, None
    if align:
        homography, signals = match_homography(target, reference, scratch_dir=scratch_dir)
        if homography is None:                      # registration failed: no prompt
            return torch.zeros(SIZE, SIZE), signals
    d = dense_distance_map(calibration.params, calibration.cx, calibration.cy,
                           homography=homography)
    # float16: the precision the benchmark prompts are stored at
    return distance_map_tensor(d.astype(np.float16), SIZE), signals


@torch.no_grad()
def predict(model, target, reference, calibration, align=True, scratch_dir=CROPS_DIR):
    """Score one photo. `model` comes from `model.load_checkpoint`; `reference`
    is the camera's stored flag photo and `calibration` its GroundPlaneFit."""
    D_R, signals = reference_prompt(target, reference, calibration, align, scratch_dir)
    W, H = Image.open(target).size
    x = network_input(load_imagenet_tensor(target, SIZE), load_imagenet_tensor(reference, SIZE),
                      D_R)[None].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(pixel_values=x).predicted_depth  # (1,3,S,S)
    q05, q50, q95 = read_prediction(pred, (H, W))
    return Prediction(q05, q50, q95, signals)
