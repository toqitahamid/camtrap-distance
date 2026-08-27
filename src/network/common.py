"""Shared constants, data loading and losses for the distance network."""
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.calibration.camera import load_cameras
from src.calibration.data import PROMPTS_DIR
from src.network.build_targets import split_of

MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
SIZE = 518
W_FLAG, W_PLANE = 1.0, 0.3
PROMPT_SCALE = 20.0  # meters -> ~[0,1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)



def distance_map_tensor(plane_d, size=SIZE):
    """Stored plane_d array -> (size, size) float tensor, NaN (no ground) -> 0."""
    z = torch.from_numpy(np.asarray(plane_d, dtype=np.float32))[None, None]
    return F.interpolate(torch.nan_to_num(z, nan=0.0), (size, size),
                         mode="nearest")[0, 0]


def load_distance_map(npz_path, size=SIZE):
    return distance_map_tensor(np.load(npz_path)["plane_d"], size)


def interval_metrics(lo, hi, y, alpha=0.10):
    """Empirical coverage, mean Winkler score and mean interval width (MPIW) of
    [lo, hi] against a nominal 1-alpha (90%) band: the paper's Table 4 columns."""
    lo, hi, y = map(lambda a: np.asarray(a, float), (lo, hi, y))
    cov = float(np.mean((y >= lo) & (y <= hi)))
    w = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return cov, float(np.mean(w)), float(np.mean(hi - lo))


def scale_invariant_log_loss(pred, gt, wt, lam=0.5, eps=1e-7):
    mask = (wt > 0) & (gt > 0.5) & (gt < 25.0)
    if mask.sum() < 10:
        return None
    w = wt[mask]
    dlog = torch.log(pred[mask].clamp(min=1e-3)) - torch.log(gt[mask])
    m = (w * dlog).sum() / w.sum()
    var = (w * dlog.pow(2)).sum() / w.sum()
    return torch.sqrt((var - lam * m * m).clamp(min=eps))


def random_drift_homography(f, cx, cy, max_deg=6.0):
    """Full-res image homography K R K^-1 for a random rotation of 0-max_deg
    degrees about a random pitch/yaw axis: synthetic camera drift."""
    deg = random.uniform(0.0, max_deg)
    phi = random.uniform(0.0, 2 * np.pi)
    axis = np.array([np.cos(phi), np.sin(phi), 0.0])
    R = cv2.Rodrigues(np.deg2rad(deg) * axis)[0]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ R @ np.linalg.inv(K)


def _load_rgb(jpg, S):
    return np.asarray(Image.open(jpg).convert("RGB").resize((S, S), Image.BILINEAR)).copy()


def _to_imagenet_tensor(rgb_u8):
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def load_imagenet_tensor(jpg, S=SIZE):
    """JPEG -> (3,S,S) ImageNet-normalised float tensor, the network's RGB input."""
    return _to_imagenet_tensor(_load_rgb(jpg, S))


def network_input(target, reference, prompt):
    """The network's 7-channel input: target RGB, reference RGB (both from
    load_imagenet_tensor) and the reference distance map D_R in metres,
    scaled by PROMPT_SCALE. Returns (7, S, S)."""
    return torch.cat([target, reference, (prompt / PROMPT_SCALE)[None]], 0)


def read_prediction(out, size):
    """Network output (1, 3, h, w) -> (3, H, W) float32 numpy in metres,
    resampled to `size` = (H, W): the [q05, q50, q95] maps."""
    return F.interpolate(out.float(), size, mode="bilinear", align_corners=True)[0].cpu().numpy()


def _stamp_flag_anchors(gt, wt, anchors, frame, S):
    """Write each flag's surveyed distance into a 3x3 patch (clipped to the frame)."""
    W, H = frame
    clip = lambda i: min(max(i, 0), S - 1)
    for u, v, fd in anchors:                      # u,v in full-res coords
        iu, iv = int(u / W * S), int(v / H * S)
        ys, xs = slice(clip(iv - 1), clip(iv + 1) + 1), slice(clip(iu - 1), clip(iu + 1) + 1)
        gt[ys, xs] = fd
        wt[ys, xs] = W_FLAG


def synthetic_pair(reference_rgb, D_R, anchors, f, cx, cy, frame, S, aligned,
                   drift=None, residual=None):
    """Re-render a flag photo under camera drift to make a training pair.

    reference_rgb (S,S,3) uint8 and D_R (S,S) are the flag photo and its
    distance map; anchors are (u, v, distance) in full-res `frame` coordinates.
    `drift` is the full-res homography (random 0-6 deg if None). Returns
    (target_rgb, gt, anchors, prompt): the drifted photo, the drifted distance
    map as ground truth, the anchors that stay in frame, and the D_R prompt,
    which for the aligned path is pre-warped with a small residual matcher
    error (random 0-0.7 deg if None) to mimic a RoMa-aligned prompt.
    """
    W, H_px = frame
    if drift is None:
        drift = random_drift_homography(f, cx, cy)
    M = np.diag([S / W, S / H_px, 1.0])
    to_size = lambda H_full: M @ H_full @ np.linalg.inv(M)   # full-res -> S x S frame
    Hs = to_size(drift)
    target_rgb = cv2.warpPerspective(reference_rgb, Hs, (S, S), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    gt = cv2.warpPerspective(D_R.numpy(), Hs, (S, S), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kept = []
    for u, v, fd in anchors:
        q = drift @ np.array([u, v, 1.0])
        uu, vv = q[0] / q[2], q[1] / q[2]
        if 0 <= uu < W and 0 <= vv < H_px:
            kept.append((uu, vv, fd))
    prompt = D_R
    if aligned:
        if residual is None:
            residual = random_drift_homography(f, cx, cy, 0.7)
        prompt = torch.from_numpy(cv2.warpPerspective(
            D_R.numpy(), to_size(residual @ drift), (S, S), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return target_rgb, gt, kept, prompt


def training_example(target_rgb, reference_rgb, prompt, gt, anchors, frame, S, augment):
    """Assemble one training example from arrays: stamp the flag anchors into
    the ground truth with weight W_FLAG (the dense plane gets W_PLANE, no-ground
    pixels 0), photometric augmentation on the target only, and a joint
    horizontal flip. Returns (input (7,S,S), gt (S,S), weights (S,S))."""
    gt = np.asarray(gt, dtype=np.float32).copy()
    wt = np.where(gt > 0, W_PLANE, 0.0).astype(np.float32)
    _stamp_flag_anchors(gt, wt, anchors, frame, S)

    tgt = torch.from_numpy(target_rgb).permute(2, 0, 1).float() / 255.0
    if augment:
        tgt = (tgt * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        tgt = tgt.pow(random.uniform(0.8, 1.25))
    img_t = (tgt - IMAGENET_MEAN) / IMAGENET_STD
    img_r = _to_imagenet_tensor(reference_rgb)
    gt_t, wt_t = torch.from_numpy(gt), torch.from_numpy(wt)

    if augment and random.random() < 0.5:  # joint horizontal flip
        img_t, img_r, gt_t, wt_t, prompt = (t.flip(-1) for t in (img_t, img_r, gt_t, wt_t, prompt))

    return network_input(img_t, img_r, prompt), gt_t, wt_t


class RefPairs(Dataset):
    """Training / validation pairs over the surveyed cameras of one split.
    Each item is a flag photo (the target) with its sibling flag photos as
    candidate references; __getitem__ makes a synthetic drifted pair or a real
    sibling pair and hands the arrays to training_example."""

    def __init__(self, split, augment, syn_prob, size=SIZE, aligned=False):
        self.size, self.augment, self.syn_prob, self.split = size, augment, syn_prob, split
        self.aligned = aligned  # D_R pre-aligned into the target frame (RoMa)
        self.items = []
        for cam in load_cameras().values():
            if split_of(cam.site) != split:
                continue
            recs = []
            for ph in cam.photos:
                cal = cam.calibration(ph)
                if cal is None:
                    continue
                recs.append(dict(jpg=cam.photo_path(ph), npz=str(cam.distance_map(ph)),
                                 prompt=str(cam.aligned_prompt(ph, PROMPTS_DIR)),
                                 f=cal.params.f, cx=cal.cx, cy=cal.cy, frame=cal.frame))
            for i, rec in enumerate(recs):
                sibs = [r for j, r in enumerate(recs) if j != i]
                if split == "val" and not sibs:
                    continue                       # deployment needs a reference
                self.items.append((rec, sibs))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        S = self.size
        rec, sibs = self.items[i]
        use_syn = self.split == "train" and (not sibs or random.random() < self.syn_prob)

        d = np.load(rec["npz"])                    # plane_d + flag anchors of the target
        anchors = [(u, v, float(fd)) for (u, v), fd in zip(d["flag_uv"], d["flag_d"])]
        if use_syn:                                # reference = rec, target = drift(rec)
            ref_rgb = _load_rgb(rec["jpg"], S)
            tgt_rgb, gt, anchors, D_R = synthetic_pair(
                ref_rgb, distance_map_tensor(d["plane_d"], S), anchors,
                rec["f"], rec["cx"], rec["cy"], rec["frame"], S, self.aligned)
        else:                                      # real: reference = sibling, target = rec
            ref = sibs[0] if self.split == "val" else random.choice(sibs)
            ref_rgb = _load_rgb(ref["jpg"], S)
            if self.aligned and Path(rec["prompt"]).exists():
                D_R = load_distance_map(rec["prompt"], S)   # RoMa-aligned into target frame
            else:
                D_R = load_distance_map(ref["npz"], S)
            tgt_rgb = _load_rgb(rec["jpg"], S)
            gt = distance_map_tensor(d["plane_d"], S).numpy()
        return training_example(tgt_rgb, ref_rgb, D_R, gt, anchors, rec["frame"], S, self.augment)
