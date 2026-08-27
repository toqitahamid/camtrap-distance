"""Alignment behaviour that the paper's numbers depend on."""
from pathlib import Path

import numpy as np
import pytest

FLAG_ROOT = Path("data/flaglabel-dataset")


def _pair_or_skip():
    """A real two-photo camera to match. RoMa needs the imagery and a GPU."""
    pytest.importorskip("romatch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("RoMa matching needs CUDA")
    for cam in sorted(p for p in FLAG_ROOT.iterdir() if p.is_dir()):
        photos = sorted(cam.glob("*.JPG"))
        if len(photos) >= 2:
            return photos[0], photos[1]
    pytest.skip("no two-photo camera found")


@pytest.mark.slow
def test_match_roma_is_deterministic():
    """RoMa samples its matches from a certainty map, so an unseeded call gives
    a different homography every run and prompts are never byte-identical.
    Downstream that moved the evaluated p90 by ~0.01 m, which is larger than the
    tolerance the paper's numbers are pinned to."""
    from src.alignment.matchers import match_roma
    a, b = _pair_or_skip()
    src1, dst1 = match_roma(str(a), str(b))
    src2, dst2 = match_roma(str(a), str(b))
    assert np.array_equal(src1, src2), "match sampling is not reproducible"
    assert np.array_equal(dst1, dst2), "match sampling is not reproducible"


@pytest.mark.slow
def test_match_roma_leaves_the_callers_rng_alone():
    """Seeding is scoped: match_roma saves and restores the RNG state, so calling
    it cannot silently reseed a training loop's stream."""
    import torch
    from src.alignment.matchers import match_roma
    a, b = _pair_or_skip()
    torch.manual_seed(1234)
    expected = torch.randn(4)
    torch.manual_seed(1234)
    match_roma(str(a), str(b))
    assert torch.equal(torch.randn(4), expected), "match_roma disturbed the caller's RNG"
