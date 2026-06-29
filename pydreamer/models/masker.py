from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .functions import *


class AuxEncoder(nn.Module):
    """Small CNN encoder shared by inverse dynamics and reward heads during pre-training."""

    def __init__(self, in_channels: int, depth: int = 32):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, depth, 4, 2, 1),      # H -> H/2
            nn.ELU(),
            nn.Conv2d(depth, depth * 2, 4, 2, 1),        # H/2 -> H/4
            nn.ELU(),
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1),    # H/4 -> H/8
            nn.ELU(),
        )
        self.out_dim = depth * 4

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        x = self.convs(x)
        return x.mean([-2, -1])  # global average pool -> (N, out_dim)


class ObservationMasker(nn.Module):
    """
    Learns a spatial soft-mask over image observations via background-augmentation
    pre-training. After pre-training, only forward() is used inside WorldModel.

    Pre-training signals (training_step):
      1. Background contrast: mask should be HIGH where pixel diff between two
         differently-backgrounded versions of the same frame is LOW (foreground).
      2. Inverse dynamics: masked images at (t, t+1) should predict action_t+1.
      3. Reward prediction: masked images at (t, t+1) should predict reward_t+1.

    WorldModel usage (forward only):
        masked_image, mask = masker(obs['image'])
        # masked_image -> encoder (and decoder reconstruction target)
        # mask         -> logged as tensor for visualization

    After pre-training the mask is frozen and also used as spatial weights
    on the decoder reconstruction loss (high mask = pixel matters more).

    Args:
        in_channels: image channels after preprocessing (e.g. 3 for RGB DMC)
        action_dim:  environment action dimension (for inverse dynamics head)
        depth:       base channel width for mask CNN and aux encoder
        actor_dist:  'tanh_normal' (continuous) or 'onehot' (discrete),
                     controls inverse dynamics loss type
    """

    def __init__(self,
                 in_channels: int,
                 action_dim: int,
                 depth: int = 32,
                 actor_dist: str = 'tanh_normal'):
        super().__init__()
        self.actor_dist = actor_dist

        # --- Mask network (used in both pre-training and WorldModel) ---
        # Same-padding convs preserve spatial resolution for any H x W
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, depth, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(depth, depth * 2, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(depth * 2, depth, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(depth, 1, 1),  # per-pixel mask logit
        )
        # Initialize bias to +2 so initial mask ≈ sigmoid(2) ≈ 0.88 (全保留).
        # Mask starts near 1 everywhere, then learns to suppress background.
        nn.init.constant_(self.mask_net[-1].bias, 2.0)

        # --- Auxiliary heads (pre-training only, discarded after) ---
        self.aux_encoder = AuxEncoder(in_channels, depth)
        aux_dim = self.aux_encoder.out_dim  # depth * 4

        # Inverse dynamics: (feat_t, feat_t+1) -> action_t
        self.inverse_head = nn.Sequential(
            nn.Linear(aux_dim * 2, aux_dim),
            nn.ELU(),
            nn.Linear(aux_dim, action_dim),
        )

        # Reward prediction: (feat_t, feat_t+1) -> reward_t+1
        # Two frames (same as inverse dynamics) so velocity-based DMC rewards
        # (cheetah/walker/quadruped run) are predictable, not just static-pose ones.
        self.reward_head = nn.Sequential(
            nn.Linear(aux_dim * 2, aux_dim),
            nn.ELU(),
            nn.Linear(aux_dim, 1),
        )

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image: (T, B, C, H, W)  preprocessed float tensor

        Returns:
            masked_image: (T, B, C, H, W)
            mask:         (T, B, 1, H, W)  in [0, 1]
        """
        T, B, C, H, W = image.shape
        x = image.reshape(T * B, C, H, W)

        mask_logits = self.mask_net(x)         # (T*B, 1, H, W)
        mask = torch.sigmoid(mask_logits)
        masked_image = x * mask                # broadcast over channels

        return masked_image.reshape(T, B, C, H, W), mask.reshape(T, B, 1, H, W)

    # ------------------------------------------------------------------
    # Pre-training interface
    # ------------------------------------------------------------------

    def training_step(self,
                      img_v1: Tensor,
                      img_v2: Tensor,
                      action: Tensor,
                      reward: Tensor,
                      reset: Tensor,
                      weight_bg: float = 1.0,
                      weight_inv: float = 1.0,
                      weight_reward: float = 1.0,
                      weight_coverage: float = 0.1,
                      mask_temp: float = 0.5,
                      ) -> Tuple[Tensor, Dict]:
        """
        Args:
            img_v1:  (T, B, C, H, W)  augmented version 1 (one background)
            img_v2:  (T, B, C, H, W)  augmented version 2 (different background,
                                       same foreground/state as img_v1)
            action:  (T, B, A)         one-hot or continuous action taken at each step
            reward:  (T, B)            reward received at each step
            reset:   (T, B)            episode boundary flags (bool/float)

        Returns:
            loss:    scalar
            metrics: dict of detached floats for logging
        """
        T, B, C, H, W = img_v1.shape

        # Apply mask (gradients flow through mask_net via all three losses)
        masked_v1, mask = self.forward(img_v1)   # mask: (T, B, 1, H, W)

        # --- Loss 1: direct mask supervision from background contrast ---
        # v1/v2 share the foreground and differ only in background, so bg_diff≈0 on
        # the foreground and large on the background. Turn that into a *spatially
        # localized* soft target and supervise the mask directly (BCE).
        #
        # Why not the old force-balance ((mask*bg_diff) down + coverage up)? A
        # multiplicative soft mask is scale-degenerate: a uniform small mask keeps
        # all information (encoder just rescales), so inverse/reward cannot localize
        # or hold the mask up, and coverage is uniform — the only localizing signal
        # is bg_diff, and under area-dominant backgrounds the CNN collapses to a
        # uniform tiny mask. A direct localized target cannot collapse to a uniform
        # value, and it teaches mask_net to predict the foreground from a single
        # image (which is all that is available at deployment time).
        with torch.no_grad():
            bg_diff = (img_v1 - img_v2).abs().mean(dim=2, keepdim=True)  # (T, B, 1, H, W)
            bg_diff = bg_diff / (bg_diff.mean() + 1e-6)                  # normalize, scale ~1
            fg_target = torch.exp(-bg_diff / mask_temp)                  # 1 on fg, ->0 on bg
        mask_c = mask.clamp(1e-4, 1 - 1e-4)
        loss_mask = F.binary_cross_entropy(mask_c, fg_target)

        # --- Loss 2: inverse dynamics ---
        # masked features at (t, t+1) should predict action_t
        tb = T * B
        feat = self.aux_encoder(masked_v1.reshape(tb, C, H, W))   # (T*B, aux_dim)
        feat = feat.reshape(T, B, -1)                               # (T, B, aux_dim)

        valid = ~reset[1:].bool()  # (T-1, B) - skip boundaries
        feat_t  = feat[:-1][valid]                                  # (N, aux_dim)
        feat_t1 = feat[1:][valid]                                   # (N, aux_dim)
        action_pred = self.inverse_head(torch.cat([feat_t, feat_t1], dim=-1))  # (N, A)

        action_target = action[1:][valid]                           # (N, A)
        if self.actor_dist == 'onehot':
            loss_inv = F.cross_entropy(action_pred, action_target.argmax(-1))
        else:
            loss_inv = F.mse_loss(action_pred, action_target)

        # --- Loss 3: reward prediction (two frames, same alignment as inverse) ---
        # reward[t+1] is the reward of the transition (o_t -> o_t+1), predicted
        # from the same (feat_t, feat_t+1) pair, with the same boundary mask.
        reward_pred = self.reward_head(torch.cat([feat_t, feat_t1], dim=-1)).squeeze(-1)  # (N,)
        reward_target = reward[1:][valid]                                                  # (N,)
        loss_reward = F.mse_loss(reward_pred, reward_target)

        # --- Total ---
        # weight_bg now weights the (supervised) mask loss. inverse/reward are
        # auxiliary: they ground the aux_encoder but cannot localize the mask, so
        # they can be set to 0 for a pure mask-localization sanity check.
        loss = (weight_bg     * loss_mask
              + weight_inv    * loss_inv
              + weight_reward * loss_reward)

        with torch.no_grad():
            # mask averaged over background vs foreground regions (bg_diff above/below
            # its own mean) — the gap is the clearest sanity signal: bg should drop,
            # fg should stay high.
            is_bg = (bg_diff > 1.0).float()  # bg_diff is normalized to mean 1
            mask_bg = (mask * is_bg).sum() / is_bg.sum().clamp(min=1.0)
            mask_fg = (mask * (1 - is_bg)).sum() / (1 - is_bg).sum().clamp(min=1.0)

        metrics = {
            'loss_masker':   loss.detach(),
            'loss_mask':     loss_mask.detach(),
            'loss_inv':      loss_inv.detach(),
            'loss_reward':   loss_reward.detach(),
            'mask_mean':     mask.detach().mean(),
            'mask_std':      mask.detach().std(),
            'mask_bg':       mask_bg,          # should -> 0 (background suppressed)
            'mask_fg':       mask_fg,          # should stay high (foreground kept)
        }
        return loss, metrics
