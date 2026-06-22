from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .functions import *
from .common import *


class MultiDecoder(nn.Module):

    def __init__(self, features_dim, conf):
        super().__init__()
        self.image_weight = conf.image_weight
        self.vecobs_weight = conf.vecobs_weight
        self.reward_weight = conf.reward_weight
        self.terminal_weight = conf.terminal_weight

        if conf.image_decoder == 'cnn':
            self.image = ConvDecoder(in_dim=features_dim,
                                     out_channels=conf.image_channels,
                                     cnn_depth=conf.cnn_depth)
        elif conf.image_decoder == 'dense':
            self.image = CatImageDecoder(in_dim=features_dim,
                                         out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                                         hidden_layers=conf.image_decoder_layers,
                                         layer_norm=conf.layer_norm,
                                         min_prob=conf.image_decoder_min_prob)
        elif not conf.image_decoder:
            self.image = None
        else:
            assert False, conf.image_decoder

        if conf.reward_decoder_categorical:
            self.reward = DenseCategoricalSupportDecoder(
                in_dim=features_dim,
                support=clip_rewards_np(conf.reward_decoder_categorical, conf.clip_rewards),  # reward_decoder_categorical values are untransformed 
                hidden_layers=conf.reward_decoder_layers,
                layer_norm=conf.layer_norm)
        else:
            self.reward = DenseNormalDecoder(in_dim=features_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)

        self.terminal = DenseBernoulliDecoder(in_dim=features_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)

        if conf.vecobs_size:
            self.vecobs = DenseNormalDecoder(in_dim=features_dim, out_dim=conf.vecobs_size, hidden_layers=4, layer_norm=conf.layer_norm)
        else:
            self.vecobs = None

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      extra_metrics: bool = False,
                      image_mask: Optional[Tensor] = None
                      ) -> Tuple[TensorTBI, Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = {}
        metrics = {}
        loss_reconstr = 0

        if self.image:
            # image_mask (spatial attention weights) only applies to the conv image decoder.
            if image_mask is not None:
                loss_image_tbi, loss_image, image_rec = self.image.training_step(features, obs['image'], mask=image_mask)
            else:
                loss_image_tbi, loss_image, image_rec = self.image.training_step(features, obs['image'])
            loss_reconstr += self.image_weight * loss_image_tbi
            metrics.update(loss_image=loss_image.detach().mean())
            tensors.update(loss_image=loss_image.detach(),
                           image_rec=image_rec.detach())

        if self.vecobs:
            loss_vecobs_tbi, loss_vecobs, vecobs_rec = self.vecobs.training_step(features, obs['vecobs'])
            loss_reconstr += self.vecobs_weight * loss_vecobs_tbi
            metrics.update(loss_vecobs=loss_vecobs.detach().mean())
            tensors.update(loss_vecobs=loss_vecobs.detach(),
                           vecobs_rec=vecobs_rec.detach())

        loss_reward_tbi, loss_reward, reward_rec = self.reward.training_step(features, obs['reward'])
        loss_reconstr += self.reward_weight * loss_reward_tbi
        metrics.update(loss_reward=loss_reward.detach().mean())
        tensors.update(loss_reward=loss_reward.detach(),
                       reward_rec=reward_rec.detach())

        loss_terminal_tbi, loss_terminal, terminal_rec = self.terminal.training_step(features, obs['terminal'])
        loss_reconstr += self.terminal_weight * loss_terminal_tbi
        metrics.update(loss_terminal=loss_terminal.detach().mean())
        tensors.update(loss_terminal=loss_terminal.detach(),
                       terminal_rec=terminal_rec.detach())

        if extra_metrics:
            if isinstance(self.reward, DenseCategoricalSupportDecoder):
                # TODO: logic should be moved to appropriate decoder
                reward_cat = self.reward.to_categorical(obs['reward'])
                for i in range(len(self.reward.support)):
                    # Logprobs for specific categorical reward values
                    mask_rewardp = reward_cat == i  # mask where categorical reward has specific value
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
                    metrics[f'loss_reward{i}'] = nanmean(loss_rewardp)  # index by support bucket, not by value
                    tensors[f'loss_reward{i}'] = loss_rewardp
            else:
                for sig in [-1, 1]:
                    # Logprobs for positive and negative rewards
                    mask_rewardp = torch.sign(obs['reward']) == sig  # mask where reward is positive or negative
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
                    metrics[f'loss_reward{sig}'] = nanmean(loss_rewardp)
                    tensors[f'loss_reward{sig}'] = loss_rewardp

            mask_terminal1 = obs['terminal'] > 0  # mask where terminal is 1
            loss_terminal1 = loss_terminal * mask_terminal1 / mask_terminal1  # set to nan where ~mask
            metrics['loss_terminal1'] = nanmean(loss_terminal1)
            tensors['loss_terminal1'] = loss_terminal1

        return loss_reconstr, metrics, tensors


class ConvDecoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_channels=3,
                 cnn_depth=32,
                 mlp_layers=0,
                 layer_norm=True,
                 activation=nn.ELU
                 ):
        super().__init__()
        self.in_dim = in_dim
        kernels = (5, 5, 6, 6)
        stride = 2
        d = cnn_depth
        if mlp_layers == 0:
            layers = [
                nn.Linear(in_dim, d * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = d * 32
            norm = nn.LayerNorm if layer_norm else NoNorm
            layers = [
                nn.Linear(in_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()]

        self.model = nn.Sequential(
            # FC
            *layers,
            nn.Unflatten(-1, (d * 32, 1, 1)),
            # Deconv
            nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(d * 2, d, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(d, out_channels, kernels[3], stride))

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        sqerr = 0.5 * torch.square(output - target)  # (B,C,H,W), per-pixel MSE
        if mask is None:
            loss = sqerr.sum(dim=[-1, -2, -3])  # sum over C,H,W
        else:
            # Spatially-weighted reconstruction: only mask-high (task-relevant) pixels
            # contribute. Normalize by mean mask value so the loss magnitude stays on the
            # same scale as the unmasked sum (mask=all-ones recovers the plain sum), which
            # keeps the KL/reconstruction balance (kl_weight) unchanged. mask is detached
            # by the caller, so this reweighting never tries to shrink the mask itself.
            mask, _ = flatten_batch(mask, 3)  # (B,1,H,W)
            wsum = (sqerr * mask).sum(dim=[-1, -2, -3])           # (B,)
            norm = mask.mean(dim=[-1, -2, -3]).clamp(min=1e-3)    # (B,)
            loss = wsum / norm
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW, mask: Optional[Tensor] = None) -> Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it
        if mask is not None:
            mask = insert_dim(mask, 2, I)  # (T,B,1,H,W) => (T,B,I,1,H,W), broadcasts over C

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target, mask)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean(dim=2)  # TBICHW => TBCHW

        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded


class CatImageDecoder(nn.Module):
    """Dense decoder for categorical image, e.g. map"""

    def __init__(self, in_dim, out_shape=(33, 7, 7), activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True, min_prob=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_shape = out_shape
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        if hidden_layers >= 1:
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
                nn.Linear(hidden_dim, np.prod(out_shape)),
                nn.Unflatten(-1, out_shape)]
        else:  
            # hidden_layers == 0
            layers += [
                nn.Linear(in_dim, np.prod(out_shape)),
                nn.Unflatten(-1, out_shape)]
        self.model = nn.Sequential(*layers)
        self.min_prob = min_prob

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)  # float(*,C,H,W) => int(*,H,W)
        assert target.dtype == torch.int64, 'Target should be categorical'
        output, bd = flatten_batch(output, len(self.out_shape))     # (*,C,H,W) => (B,C,H,W)
        target, _ = flatten_batch(target, len(self.out_shape) - 1)  # (*,H,W) => (B,H,W)

        if self.min_prob == 0:
            loss = F.nll_loss(F.log_softmax(output, 1), target, reduction='none')  # = F.cross_entropy()
        else:
            prob = F.softmax(output, 1)
            prob = (1.0 - self.min_prob) * prob + self.min_prob * (1.0 / prob.size(1))  # mix with uniform prob
            loss = F.nll_loss(prob.log(), target, reduction='none')

        if len(self.out_shape) == 3:
            loss = loss.sum(dim=[-1, -2])  # (*,H,W) => (*)
        assert len(loss.shape) == 1
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW) -> Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        logits = self.forward(features)
        loss_tbi = self.loss(logits, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB

        assert len(logits.shape) == 6   # TBICHW
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)  # normalize C
        logits = torch.logsumexp(logits, dim=2)  # aggregate I => TBCHW
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)  # normalize C again
        decoded = logits

        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded


class DenseBernoulliDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.model = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Bernoulli(logits=y.float())
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded


class DenseNormalDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, layer_norm=True, std=0.3989422804):
        super().__init__()
        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        var = self.std ** 2  # var cancels denominator, which makes loss = 0.5 (target-output)^2
        return -output.log_prob(target) * var

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, Tensor]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return loss_tbi, loss_tb, decoded


class DenseCategoricalSupportDecoder(nn.Module):
    """
    Represent continuous variable distribution by discrete set of support values.
    Useful for reward head, which can be e.g. [-10, 0, 1, 10]
    """

    def __init__(self, in_dim, support=[0.0, 1.0], hidden_dim=400, hidden_layers=2, layer_norm=True):
        assert isinstance(support, (list, np.ndarray))
        super().__init__()
        self.model = MLP(in_dim, len(support), hidden_dim, hidden_layers, layer_norm)
        self.support = np.array(support).astype(float)
        self._support = nn.Parameter(torch.tensor(support).to(torch.float), requires_grad=False)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = CategoricalSupport(logits=y.float(), sup=self._support.data)
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        target = self.to_categorical(target)
        return -output.log_prob(target)

    def to_categorical(self, target: Tensor) -> Tensor:
        # TODO: should interpolate between adjacent values, like in MuZero
        distances = torch.square(target.unsqueeze(-1) - self._support)
        return distances.argmin(-1)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded


class InverseDynamicsHead(nn.Module):
    """Inverse dynamics head: predict the action executed between two consecutive
    *encoder embeddings* e_t and e_{t+1}.

    Design constraints (DreamerV2 inverse-dynamics auxiliary loss):
      - Input is the raw encoder embedding e_t = encoder(o_t), NOT the RSSM latent.
        Using the RSSM state would leak the action (h_{t+1}=f(h_t,z_t,a_t)) and make
        this head degenerate.
      - Gradients flow back into the encoder (embed is not detached), so the auxiliary
        loss shapes the encoder representation against visual distractors.
      - Time alignment follows this codebase's convention `action[t]` = action that
        led to obs[t] (see data.py: `action[i] -> obs[i] -> action_next[i] -> obs[i+1]`).
        So the action between e_t and e_{t+1} is `action[t+1]`, i.e. target = action[1:].
    """

    def __init__(self, embed_dim, action_dim, action_dist='onehot',
                 hidden_dim=400, hidden_layers=2, layer_norm=True, activation=nn.ELU):
        super().__init__()
        self.action_dist = action_dist
        self.action_dim = action_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        # Build the MLP manually (not common.MLP) to avoid its out_dim==1 Flatten
        # special-case, which would break shapes for 1-D continuous actions (e.g. DMC cartpole).
        layers = []
        dim = 2 * embed_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(dim, hidden_dim), norm(hidden_dim, eps=1e-3), activation()]
            dim = hidden_dim
        layers += [nn.Linear(dim, action_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, embed: Tensor) -> Tensor:
        # embed: (T,B,E) => predict action for each adjacent pair (T-1,B,A)
        pair = torch.cat([embed[:-1], embed[1:]], dim=-1)  # (T-1,B,2E)
        return self.model(pair)

    def training_step(self, embed: Tensor, action: Tensor, reset: Tensor) -> Tuple[Tensor, Tensor]:
        # embed: (T,B,E)  action: (T,B,A)  reset: (T,B)
        assert len(embed.shape) == 3, f'Expected encoder embed (T,B,E), got {embed.shape}'
        T = embed.shape[0]
        assert action.shape[0] == T

        pred = self.forward(embed)  # (T-1,B,A), aligned to predict action[1:]
        target = action[1:]         # (T-1,B,A): action executed between e_t and e_{t+1}

        # Mask transitions crossing an episode boundary: reset[t+1]=True means o_{t+1}
        # starts a new episode, so the (e_t, e_{t+1}) pair (and action[t+1]) is invalid.
        mask = (reset[1:] <= 0).to(pred.dtype)  # (T-1,B)

        if self.action_dist == 'onehot':
            logits = pred.reshape(-1, self.action_dim)
            tgt = target.reshape(-1, self.action_dim).argmax(dim=-1)
            loss_step = F.cross_entropy(logits, tgt, reduction='none').reshape(target.shape[:-1])  # (T-1,B)
        else:
            # Mean over action dims (not sum) so inverse_head_weight is invariant to
            # action_dim across envs. For A=1 (e.g. cartpole) mean==sum, so no change there.
            loss_step = torch.square(pred - target).mean(dim=-1)  # (T-1,B), L2 per step

        denom = mask.sum().clamp(min=1.0)
        loss = (loss_step * mask).sum() / denom
        return loss, loss_step.detach()
