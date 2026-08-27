"""The training-data rules, checked on synthetic arrays without the dataset."""
import numpy as np
import pytest
import torch

import src.network.common as common
from src.network.common import (PROMPT_SCALE, W_FLAG, W_PLANE, synthetic_pair, training_example)

S, FRAME = 64, (640, 360)
IDENTITY = np.eye(3)


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (S, S, 3), dtype=np.uint8)
    D_R = torch.from_numpy(np.tile(np.linspace(0, 16, S, dtype=np.float32), (S, 1)))  # 0 = no ground
    anchors = [(320.0, 180.0, 7.0), (10.0, 340.0, 3.0)]
    return rgb, D_R, anchors


def test_identity_drift_gives_the_reference_back(reference):
    rgb, D_R, anchors = reference
    tgt, gt, kept, prompt = synthetic_pair(rgb, D_R, anchors, 3000, 320, 180, FRAME, S,
                                           aligned=False, drift=IDENTITY)
    assert np.array_equal(tgt, rgb) and np.array_equal(gt, D_R.numpy()) and kept == anchors
    assert torch.equal(prompt, D_R)


def test_anchors_that_drift_out_of_frame_are_dropped(reference):
    rgb, D_R, anchors = reference
    shift = np.array([[1, 0, 400.0], [0, 1, 0], [0, 0, 1]])   # pushes u=320 past the 640 edge
    _, _, kept, _ = synthetic_pair(rgb, D_R, anchors, 3000, 320, 180, FRAME, S,
                                   aligned=False, drift=shift)
    assert [a[2] for a in kept] == [3.0]


def test_aligned_prompt_is_prewarped_with_the_residual(reference):
    rgb, D_R, anchors = reference
    _, _, _, prompt = synthetic_pair(rgb, D_R, anchors, 3000, 320, 180, FRAME, S, aligned=True,
                                     drift=IDENTITY, residual=IDENTITY)
    assert torch.equal(prompt, D_R)


def test_weights_and_anchor_stamp_use_the_frame(reference):
    rgb, D_R, anchors = reference
    x, gt, wt = training_example(rgb, rgb, D_R, D_R.numpy(), anchors, FRAME, S, augment=False)
    assert x.shape == (7, S, S) and torch.equal(x[6], D_R / PROMPT_SCALE)
    iu, iv = int(320 / 640 * S), int(180 / 360 * S)                 # anchor in frame coordinates
    assert (wt[iv - 1:iv + 2, iu - 1:iu + 2] == W_FLAG).all() and (gt[iv, iu] == 7.0)
    assert wt[0, 0] == 0.0 and gt[0, 0] == 0.0                       # no ground, no weight
    assert wt[5, 50] == W_PLANE                                       # dense plane, no anchor


def test_flip_moves_all_five_tensors_together(reference, monkeypatch):
    rgb, D_R, anchors = reference
    monkeypatch.setattr(common.random, "uniform", lambda a, b: 1.0 if b > 1 else 0.0)  # no-op photometrics
    monkeypatch.setattr(common.random, "random", lambda: 0.0)                             # force the flip
    x, gt, wt = training_example(rgb, rgb, D_R, D_R.numpy(), anchors, FRAME, S, augment=True)
    x0, gt0, wt0 = training_example(rgb, rgb, D_R, D_R.numpy(), anchors, FRAME, S, augment=False)
    assert torch.equal(x, x0.flip(-1)) and torch.equal(gt, gt0.flip(-1)) and torch.equal(wt, wt0.flip(-1))
