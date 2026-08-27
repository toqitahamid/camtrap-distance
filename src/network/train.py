#!/usr/bin/env python3
"""Train the distance network: metric distance plus a monotone 90% interval.

Single phase from base Depth Anything V2: encoder frozen for the first
--unfreeze-frac of epochs (adapter, neck and heads train), then the encoder
unfreezes at --encoder-lr. The interval loss is detached from the trunk (see
model.QuantileHead), so it trains only the offset heads. Checkpoint selection
is on validation MAE at flag pixels. The camera split is the one in
build_targets.py.

Usage:
  PYTHONPATH=. python src/network/train.py --aligned-ref --epochs 100 --seed 1 --out outputs/ckpt
  PYTHONPATH=. python src/network/train.py --selfcheck      # CPU sanity, no GPU
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.network.common import scale_invariant_log_loss, W_FLAG, RefPairs
from src.network.model import build_model, save_checkpoint, quantile_loss_log

TAU_LO, TAU_HI = 0.05, 0.95


@torch.no_grad()
def val_point_mae(model, batches, device):
    """MAE of the q50 channel at flag pixels over pre-built validation batches."""
    model.eval()
    errs = []
    for x, gt, wt in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(pixel_values=x.to(device)).predicted_depth   # (B,3,H,W)
        pred = F.interpolate(pred.float(), gt.shape[-2:], mode="bilinear",
                             align_corners=True)[:, 1].cpu()           # q50
        m = (wt == W_FLAG) & (gt > 0.5)
        if m.any():
            errs.append((pred[m] - gt[m]).abs().mean().item())
    model.train()
    return sum(errs) / max(len(errs), 1)


def selfcheck():
    """CPU-only assert-based sanity: (1) pinball(log) recovers ordered quantiles;
    (2) tiny model builds, forwards 7ch->3ch, is monotone q05<q50<q95, and the
    interval loss leaves the trunk (base conv3) with zero gradient."""
    import torch.nn as nn
    from transformers import DepthAnythingConfig, AutoModelForDepthEstimation
    from src.network.model import widen_input_to_7ch, QuantileHead

    # (1) pinball ordering: minimizing pinball over a constant recovers the tau-quantile
    rng = np.random.default_rng(0)
    y = torch.from_numpy(rng.lognormal(1.5, 0.5, size=2000).astype(np.float32)).clamp(0.6, 24)
    wt = torch.ones_like(y)
    cand = torch.linspace(0.7, 23.0, 400)
    best = {}
    for tau in (TAU_LO, 0.5, TAU_HI):
        losses = [quantile_loss_log(c.expand_as(y), y, wt, tau).item() for c in cand]
        best[tau] = float(cand[int(np.argmin(losses))])
    assert best[TAU_LO] < best[0.5] < best[TAU_HI], f"pinball ordering broken: {best}"
    true_q = {t: float(torch.quantile(y, t)) for t in (TAU_LO, 0.5, TAU_HI)}
    for t in best:
        assert abs(best[t] - true_q[t]) < 1.5, f"pinball tau={t} off: {best[t]} vs {true_q[t]}"
    print(f"[self-check] pinball-log quantiles {(  {k: round(v,2) for k,v in best.items()})} "
          f"~ empirical {({k: round(v,2) for k,v in true_q.items()})}  (ordered, OK)")

    # (2) tiny model build + forward 7ch -> 3ch, monotone, trunk-isolated interval loss
    bb = dict(model_type="dinov2", hidden_size=32, num_hidden_layers=4,
              num_attention_heads=2, image_size=518, patch_size=14,
              out_indices=[1, 2, 3, 4], reshape_hidden_states=False)
    cfg = DepthAnythingConfig(backbone_config=bb, patch_size=14, reassemble_hidden_size=32,
                              neck_hidden_sizes=[8, 8, 8, 8], fusion_hidden_size=16,
                              head_hidden_size=8, depth_estimation_type="metric", max_depth=80)
    m = widen_input_to_7ch(AutoModelForDepthEstimation.from_config(cfg))
    m.head = QuantileHead(m.head)
    x = torch.randn(1, 7, 518, 518)
    out = m(pixel_values=x).predicted_depth
    assert out.shape == (1, 3, 518, 518), f"expected (1,3,518,518), got {tuple(out.shape)}"
    q05, q50, q95 = out[:, 0], out[:, 1], out[:, 2]
    assert bool((q05 < q50).all() and (q50 < q95).all()), "monotonicity q05<q50<q95 violated"
    # interval loss must not touch the trunk (base conv3)
    m.zero_grad()
    (F.softplus(out[:, 2]).mean() + F.softplus(-out[:, 0]).mean()).backward()
    g = m.head.base.conv3.weight.grad
    assert g is not None and float(g.abs().sum()) == 0.0, "interval loss leaked into trunk conv3"
    assert m.head.head_hi.weight.grad is not None, "delta head got no gradient"
    print(f"[self-check] build 7ch->3ch {tuple(out.shape)} | monotone OK | "
          f"trunk conv3 grad={float(g.abs().sum()):.1f} (isolated) | delta heads trained  (OK)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/ckpt")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--syn-prob", type=float, default=0.6)
    ap.add_argument("--aligned-ref", action="store_true",
                    help="D_R pre-aligned via RoMa (prompts_roma) — paper condition")
    ap.add_argument("--head-lr", type=float, default=5e-5)
    ap.add_argument("--encoder-lr", type=float, default=5e-6,
                    help="encoder LR after unfreeze")
    ap.add_argument("--unfreeze-frac", type=float, default=0.4,
                    help="fraction of epochs the encoder stays frozen")
    ap.add_argument("--selfcheck", action="store_true", help="CPU sanity, no GPU")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda"
    model = build_model(device)

    # ---- param groups & freeze policy: encoder frozen for the first unfreeze_frac ----
    is_enc = lambda n: n.startswith("backbone") and "patch_embeddings" not in n
    enc = [p for n, p in model.named_parameters() if is_enc(n)]
    other = [p for n, p in model.named_parameters() if not is_enc(n)]
    for p in enc:
        p.requires_grad = False
    opt = torch.optim.AdamW([{"params": other, "lr": a.head_lr}, {"params": enc, "lr": a.encoder_lr}],
                            weight_decay=0.01)
    unfreeze_ep = int(a.unfreeze_frac * a.epochs)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"encoder frozen<=ep{unfreeze_ep}, then lr {a.encoder_lr} | trainable {n_tr:.2f}M", flush=True)

    train_ds = RefPairs("train", True, a.syn_prob, a.size, aligned=a.aligned_ref)
    val_ds = RefPairs("val", False, 0.0, a.size, aligned=a.aligned_ref)
    print(f"train {len(train_ds)}, val(deploy) {len(val_ds)}", flush=True)
    val_batches = list(DataLoader(val_ds, batch_size=4))   # deterministic: build once
    loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (1 - it / (a.epochs * len(loader))) ** 0.9)

    best, log = float("inf"), []
    ckdir = Path(a.out)
    model.train()
    for ep in range(1, a.epochs + 1):
        if ep == unfreeze_ep + 1:
            for p in enc:
                p.requires_grad = True         # AdamW starts updating the encoder now
            print(f"epoch {ep}: unfroze encoder @ lr {a.encoder_lr}", flush=True)
        tot, tp, npin, nb = 0.0, 0.0, 0.0, 0
        for x, gt, wt in loader:
            x, gt, wt = x.to(device), gt.to(device), wt.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(pixel_values=x).predicted_depth              # (B,3,H,W)
            pred = F.interpolate(pred.float(), gt.shape[-2:], mode="bilinear",
                                 align_corners=True)
            q05, q50, q95 = pred[:, 0], pred[:, 1], pred[:, 2]
            l_point = scale_invariant_log_loss(q50, gt, wt)                                  # trunk (0 grad if frozen)
            l_lo = quantile_loss_log(q05, gt, wt, TAU_LO)
            l_hi = quantile_loss_log(q95, gt, wt, TAU_HI)
            pin = [t for t in (l_lo, l_hi) if t is not None]
            if l_point is None and not pin:
                continue
            loss = 0.0
            if l_point is not None:
                loss = loss + l_point
            if pin:
                loss = loss + sum(pin)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss); tp += float(l_point) if l_point is not None else 0.0
            npin += float(sum(pin)) if pin else 0.0; nb += 1
        vmae = val_point_mae(model, val_batches, device)
        log.append(dict(epoch=ep, train_loss=round(tot / max(nb, 1), 4),
                        train_point=round(tp / max(nb, 1), 4),
                        train_pin=round(npin / max(nb, 1), 4),
                        val_point=round(vmae, 3)))
        print(f"epoch {ep}: loss {tot/max(nb,1):.4f} (point {tp/max(nb,1):.4f} "
              f"pin {npin/max(nb,1):.4f}) | val_point {vmae:.3f}", flush=True)
        if vmae < best:                        # select on validation MAE
            best = vmae
            save_checkpoint(model, ckdir / "best")
    save_checkpoint(model, ckdir / "last")
    with open(ckdir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader(); w.writerows(log)
    print(f"best val point-median MAE {best:.3f} m | saved -> {ckdir}", flush=True)


if __name__ == "__main__":
    main()
