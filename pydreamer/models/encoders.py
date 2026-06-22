import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .functions import *
from .common import *


def gumbel_sigmoid(logits: Tensor, tau: float, eps: float = 1e-8) -> Tensor:
    """Binary-Concrete (relaxed Bernoulli) sample of a gate with logit `logits`."""
    u = torch.rand_like(logits)
    logistic = torch.log(u + eps) - torch.log1p(-u + eps)  # = log(u) - log(1-u)
    return torch.sigmoid((logits + logistic) / tau)


def bernoulli_kl(q: Tensor, p: float, eps: float = 1e-6) -> Tensor:
    """KL( Bernoulli(q) || Bernoulli(p) ), elementwise. p is the (scalar) sparse prior."""
    q = q.clamp(eps, 1 - eps)
    return q * (torch.log(q) - math.log(p)) + (1 - q) * (torch.log1p(-q) - math.log(1 - p))


class MultiEncoder(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.reward_input = conf.reward_input
        if conf.reward_input:
            encoder_channels = conf.image_channels + 2  # + reward, terminal
        else:
            encoder_channels = conf.image_channels

        if conf.image_encoder == 'cnn':
            self.encoder_image = ConvEncoder(in_channels=encoder_channels,
                                             cnn_depth=conf.cnn_depth,
                                             use_spatial_attention=getattr(conf, 'use_spatial_attention', False),
                                             gate_tap=getattr(conf, 'attention_tap', 2),
                                             mask_prior=getattr(conf, 'mask_prior', 0.1),
                                             mask_temp=getattr(conf, 'mask_temp', 0.5))
        elif conf.image_encoder == 'dense':
            self.encoder_image = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
                                              out_dim=256,
                                              hidden_layers=conf.image_encoder_layers,
                                              layer_norm=conf.layer_norm)
        elif not conf.image_encoder:
            self.encoder_image = None
        else:
            assert False, conf.image_encoder

        if conf.vecobs_size:
            self.encoder_vecobs = MLP(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        else:
            self.encoder_vecobs = None

        assert self.encoder_image or self.encoder_vecobs, "Either image_encoder or vecobs_size should be set"
        self.out_dim = ((self.encoder_image.out_dim if self.encoder_image else 0) +
                        (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0))

    def forward(self, obs: Dict[str, Tensor], gate_active: bool = True):
        # TODO:
        #  1) Make this more generic, e.g. working without image input or without vecobs
        #  2) Treat all inputs equally, adding everything via linear layer to embed_dim
        # gate_active=False bypasses the VIB gate (used while the rate weight beta=0).
        # Returns (embed, mask, rate):
        #   mask (T,B,1,h,w) = VIB gate probabilities in (0,1), for recon-weighting / viz
        #   rate (T,B)       = IB rate term KL(gate || prior), to be added to the encoder loss
        # Both are None when spatial attention is disabled.

        embeds = []
        mask = None
        rate = None

        if self.encoder_image:
            image = obs['image']
            T, B, C, H, W = image.shape
            if self.reward_input:
                reward = obs['reward']
                terminal = obs['terminal']
                reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                image = torch.cat([image,  # (T,B,C+2,H,W)
                                reward_plane.to(image.dtype),
                                terminal_plane.to(image.dtype)], dim=-3)

            out = self.encoder_image.forward(image, gate_active)  # (T,B,E) or ((T,B,E), mask, rate)
            if isinstance(out, tuple):
                embed_image, mask, rate = out
            else:
                embed_image = out
            embeds.append(embed_image)

        if self.encoder_vecobs:
            embed_vecobs = self.encoder_vecobs(obs['vecobs'])
            embeds.append(embed_vecobs)

        embed = torch.cat(embeds, dim=-1)  # (T,B,E+256)
        return embed, mask, rate


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU,
                 use_spatial_attention=False, gate_tap=2, mask_prior=0.1, mask_temp=0.5):
        super().__init__()
        self.out_dim = cnn_depth * 32
        self.use_spatial_attention = use_spatial_attention
        self.mask_prior = mask_prior
        self.mask_temp = mask_temp
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth
        out_channels = (d, d * 2, d * 4, d * 8)  # spatial: 31, 14, 6, 2

        # 4 conv+activation blocks. The VIB spatial gate is applied after `gate_tap` of them:
        #   gate_tap=1 -> 31x31, =2 -> 14x14, =3 -> 6x6, =4 -> 2x2.
        conv = []
        cin = in_channels
        for cout, k in zip(out_channels, kernels):
            conv += [nn.Conv2d(cin, cout, k, stride), activation()]
            cin = cout
        if use_spatial_attention:
            assert 1 <= gate_tap <= 4, gate_tap
            self.gate_tap = gate_tap
            self.early = nn.Sequential(*conv[:2 * gate_tap])  # up to and incl. the gate layer
            self.late = nn.Sequential(*conv[2 * gate_tap:])   # remaining convs (identity if empty)
            gate_channels = out_channels[gate_tap - 1]
            # 1x1 conv -> per-location gate logit. One Bernoulli gate per spatial cell,
            # shared across channels (a spatial attention gate).
            self.attn = nn.Conv2d(gate_channels, 1, kernel_size=1)
        else:
            self.attn = None
            self.model = nn.Sequential(*conv)
        self.flatten = nn.Flatten()

    def forward(self, x, gate_active=True):
        x, bd = flatten_batch(x, 3)
        if self.attn is None:
            embed = self.flatten(self.model(x))
            return unflatten_batch(embed, bd)

        feat = self.early(x)                       # (B, gate_ch, hs, ws)
        logits = self.attn(feat)                   # (B, 1, hs, ws)  gate logits
        g = torch.sigmoid(logits)                  # gate probability (deterministic; for recon/viz)
        if gate_active:
            if self.training:
                m = gumbel_sigmoid(logits, self.mask_temp)  # relaxed Bernoulli sample
            else:
                m = g                              # eval: expected gate, deterministic
            feat = feat * m                        # spatial gate, broadcasts over channels
        else:
            # Gate bypassed (beta=0 warmup): clean full-pass embedding, attn head frozen
            # (no gradient reaches it), so inverse dynamics learns on an undisturbed embedding.
            g = g.detach()
        embed = self.flatten(self.late(feat))      # (B, d*32)

        # IB rate term: KL(gate || sparse prior), mean over spatial positions (resolution-invariant)
        rate = bernoulli_kl(g, self.mask_prior).mean(dim=[-3, -2, -1])  # (B,)

        embed = unflatten_batch(embed, bd)
        g = unflatten_batch(g, bd)                 # (T,B,1,hs,ws)
        rate = unflatten_batch(rate, bd)           # (T,B)
        return embed, g, rate


class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = [nn.Flatten()]
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x, gate_active=True):  # gate_active ignored (no spatial gate on dense)
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y
