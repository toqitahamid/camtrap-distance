#!/usr/bin/env python3
"""Reference-conditioned Depth Anything V2 with a monotone quantile head.

Input is 7 channels (target RGB, reference RGB, reference distance map D_R).
Output `model(pixel_values=x).predicted_depth` is (B, 3, H, W) = [q05, q50, q95]
in metres. q50 is the base metric head; q05/q95 are log-space offsets through
softplus, so q05 < q50 < q95 by construction. The offset heads read detached
features, so the interval loss never moves the point estimate.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.network.common import MODEL as BASE_MODEL


def widen_input_to_7ch(model):
    """3->7 input channels on the DINOv2 patch embed: target RGB keeps the
    pretrained weights, reference RGB + D_R channels start at zero."""
    pe = model.backbone.embeddings.patch_embeddings
    old = pe.projection
    new = nn.Conv2d(7, old.out_channels, old.kernel_size, old.stride)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight
        new.bias.copy_(old.bias)
    pe.projection = new
    pe.num_channels = 7
    model.config.backbone_config.num_channels = 7
    return model


class QuantileHead(nn.Module):
    """Wraps the original DepthAnythingDepthEstimationHead. Reproduces its
    conv1->interp->conv2->relu to expose penultimate features `feat`, keeps
    conv3 for the metric median q50, and adds two softplus-delta heads for a
    monotone log-space interval. Returns stack([q05, q50, q95], dim=1)."""

    def __init__(self, base):
        super().__init__()
        self.base = base                       # keeps head.base.conv{1,2,3} names
        hc = base.conv3.in_channels            # head_hidden_size (32)
        self.head_lo = nn.Conv2d(hc, 1, kernel_size=1)
        self.head_hi = nn.Conv2d(hc, 1, kernel_size=1)
        for h in (self.head_lo, self.head_hi):  # zero-init: init interval = q50*[0.5, 2]
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def _features(self, hidden_states, patch_height, patch_width):
        b = self.base
        x = b.conv1(hidden_states[b.head_in_index])
        x = F.interpolate(x, (int(patch_height * b.patch_size),
                              int(patch_width * b.patch_size)),
                          mode="bilinear", align_corners=True)
        x = b.conv2(x)
        x = b.activation1(x)                   # (B, head_hidden, H, W)
        return x

    def forward(self, hidden_states, patch_height, patch_width):
        b = self.base
        feat = self._features(hidden_states, patch_height, patch_width)
        q50 = (b.activation2(b.conv3(feat)) * b.max_depth).squeeze(1)  # (B,H,W)
        m = torch.log(q50.clamp(min=1e-3)).detach()   # isolate interval from trunk
        fd = feat.detach()
        q05 = torch.exp(m - F.softplus(self.head_lo(fd)).squeeze(1))
        q95 = torch.exp(m + F.softplus(self.head_hi(fd)).squeeze(1))
        return torch.stack([q05, q50, q95], dim=1)     # (B, 3, H, W)


def build_model(device="cuda"):
    """Base Depth Anything V2 (metric outdoor) widened to 7 input channels and
    wrapped with the quantile head: the paper's starting point for training."""
    from transformers import AutoModelForDepthEstimation
    model = widen_input_to_7ch(AutoModelForDepthEstimation.from_pretrained(BASE_MODEL))
    model.head = QuantileHead(model.head)
    return model.to(device)


def save_checkpoint(model, ckdir):
    """save_pretrained (config + safetensors) keeps the wrapped param names
    (head.base.*, head.head_lo, head.head_hi); load_checkpoint rebuilds the exact
    architecture so the round-trip is strict."""
    ckdir = Path(ckdir)
    ckdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckdir)


def load_checkpoint(ckdir, device="cuda"):
    """Rebuild the wrapped architecture from the saved config (num_channels=7 =>
    7-ch patch embed built directly) then load the state dict strictly."""
    from transformers import AutoModelForDepthEstimation, AutoConfig
    from safetensors.torch import load_file
    ckdir = Path(ckdir)
    cfg = AutoConfig.from_pretrained(ckdir)
    model = AutoModelForDepthEstimation.from_config(cfg)  # config says 7-ch -> 7-ch conv
    model.head = QuantileHead(model.head)
    sd = load_file(ckdir / "model.safetensors")
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def quantile_loss_log(q_pred, gt, wt, tau, eps=1e-3):
    """Weighted pinball (quantile) loss in LOG-distance space (same space as
    scale_invariant_log_loss) so far targets don't dominate. L_tau(u)=max(tau*u,(tau-1)*u),
    u = log gt - log q_pred. Masked like scale_invariant_log_loss. Returns None if too few pixels."""
    mask = (wt > 0) & (gt > 0.5) & (gt < 25.0)
    if mask.sum() < 10:
        return None
    w = wt[mask]
    u = torch.log(gt[mask]) - torch.log(q_pred[mask].clamp(min=eps))
    loss = torch.maximum(tau * u, (tau - 1.0) * u)
    return (w * loss).sum() / w.sum()
