# -*- coding: utf-8 -*-
"""Losses to improve separation between confusing classes.

This file is designed to be *drop-in* with your existing train loops that call:

    loss = criterion(logits, targets)

and expect `logits` to be a Tensor of shape [B, C].

Key idea (for your R1-2 concern):
- Keep standard CE for overall 3-class performance.
- Add an explicit *pairwise margin* term to push apart the logits of two
  confusing classes (e.g., man-made vs excavation).
- Optionally use OHEM (online hard example mining) **only** on those two
  confusing classes so training focuses on the difficult boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeparationLossConfig:
    """Config for SeparationLoss.

    class_a / class_b:
        The two classes you want to separate more aggressively.
        For your dataset naming convention:
            0 = wind, 1 = manual (man-made), 2 = digger (excavation)
    """

    # CE settings
    label_smoothing: float = 0.0
    class_weights: Optional[torch.Tensor] = None  # shape [C]

    # OHEM settings (applied ONLY on class_a/class_b samples)
    ohem_ratio: float = 1.0  # 1.0 = disabled; otherwise keep top-k hardest in (0,1]

    # Pairwise margin settings (applied ONLY on class_a/class_b samples)
    margin: float = 0.5
    margin_lambda: float = 0.2
    class_a: int = 1
    class_b: int = 2


class SeparationLoss(nn.Module):
    """CrossEntropy + (optional) OHEM + (optional) pairwise margin term.

    - CE keeps global classification stable.
    - OHEM encourages the model to spend capacity on the hardest man/exc samples.
    - Margin explicitly enforces logit separation between the confusing pair.

    This is intentionally implemented without requiring access to intermediate
    features, so you can use it with your current train_VAL_* functions.
    """

    def __init__(self, cfg: SeparationLossConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.ohem_ratio <= 0 or cfg.ohem_ratio > 1:
            raise ValueError(f"ohem_ratio must be in (0, 1], got {cfg.ohem_ratio}")
        if cfg.label_smoothing < 0 or cfg.label_smoothing >= 1:
            raise ValueError(
                f"label_smoothing must be in [0,1), got {cfg.label_smoothing}"
            )
        if cfg.margin < 0:
            raise ValueError(f"margin must be >= 0, got {cfg.margin}")
        if cfg.margin_lambda < 0:
            raise ValueError(f"margin_lambda must be >= 0, got {cfg.margin_lambda}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(logits):
            raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B,C], got {tuple(logits.shape)}")
        if target.ndim != 1:
            target = target.view(-1)

        # -------------------------
        # 1) CE (optionally OHEM)
        # -------------------------
        ce_per_sample = F.cross_entropy(
            logits,
            target,
            weight=self.cfg.class_weights,
            label_smoothing=self.cfg.label_smoothing,
            reduction="none",
        )

        if self.cfg.ohem_ratio < 1.0:
            # Apply OHEM ONLY on the confusing pair, keep other classes untouched.
            mask_pair = (target == self.cfg.class_a) | (target == self.cfg.class_b)
            ce_other = ce_per_sample[~mask_pair]
            ce_pair = ce_per_sample[mask_pair]

            if ce_pair.numel() > 0:
                k = max(1, int(self.cfg.ohem_ratio * ce_pair.numel()))
                hard_pair, _ = torch.topk(ce_pair, k=k, largest=True)
                ce_used = (
                    torch.cat([ce_other, hard_pair], dim=0)
                    if ce_other.numel() > 0
                    else hard_pair
                )
            else:
                ce_used = ce_per_sample
            loss_ce = ce_used.mean()
        else:
            loss_ce = ce_per_sample.mean()

        # -------------------------
        # 2) Pairwise margin term
        # -------------------------
        if self.cfg.margin_lambda > 0:
            mask_pair = (target == self.cfg.class_a) | (target == self.cfg.class_b)
            if mask_pair.any():
                lp = logits[mask_pair]
                tp = target[mask_pair]

                logit_a = lp[:, self.cfg.class_a]
                logit_b = lp[:, self.cfg.class_b]

                # diff = (true_logit - other_logit)
                diff = torch.where(tp == self.cfg.class_a, logit_a - logit_b, logit_b - logit_a)

                # Want diff >= margin => penalty when margin - diff > 0
                loss_margin = F.relu(self.cfg.margin - diff).mean()
            else:
                loss_margin = logits.new_tensor(0.0)
        else:
            loss_margin = logits.new_tensor(0.0)

        return loss_ce + self.cfg.margin_lambda * loss_margin


def make_inverse_freq_weights(counts: list[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Create simple inverse-frequency class weights (normalized to mean=1).

    Args:
        counts: samples per class, e.g., [n_wind, n_manual, n_digger]
    """
    if len(counts) == 0:
        raise ValueError("counts must be non-empty")
    c = torch.tensor(counts, dtype=torch.float32)
    c = torch.clamp(c, min=1.0)
    w = c.sum() / c
    w = w / w.mean()  # normalize so avg weight is ~1
    if device is not None:
        w = w.to(device)
    return w
