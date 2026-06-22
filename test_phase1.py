"""
Phase-1 mask inspection: load a (warm-up) checkpoint, run a short rollout with a
random policy, compute the spatial-attention mask for every frame, and dump an npz
where the `image_rec` column is the obs+mask overlay. The unchanged `visualize.py`
then renders it as [real | overlay].

Usage:
    # ~15s of video at fps=10 (=150 frames)
    python test_phase1.py --configs defaults dmc distract \
        --use_spatial_attention True --use_inverse_head True \
        --checkpoint mlruns/0/<run_id>/artifacts/checkpoints/latest.pt \
        --out phase1_mask.npz --num_steps 150

    python visualize.py phase1_mask.npz --out phase1_mask.mp4 --fps 10
    # Columns shown:  real | reconstruct(=obs+mask overlay)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from pydreamer.envs import create_env
from pydreamer.models import Dreamer
from pydreamer.models.functions import map_structure
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import read_yamls


def _bool(x):
    return str(x).lower() in ('1', 'true', 'yes', 'y')


def build_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='phase1_mask.npz')
    parser.add_argument('--num_steps', type=int, default=150, help='frames to record (~15s at fps=10)')
    parser.add_argument('--alpha', type=float, default=0.6, help='overlay strength of the mask tint')
    parser.add_argument('--seed', type=int, default=0)
    args, remaining = parser.parse_known_args()

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        for n in name.split(','):
            conf.update(configs[n])

    parser2 = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _bool
        parser2.add_argument(f'--{key}', type=type_, default=value)
    conf = parser2.parse_args(remaining)
    return args, conf


def encoder_mask(model: Dreamer, obs_model, H, W):
    """Run the encoder and return the per-pixel mask (H,W) in [0,1], or None if disabled."""
    with torch.no_grad():
        embed, mask_feat, _ = model.wm.encoder(obs_model)  # mask_feat (1,1,1,h,w) or None
    if mask_feat is None:
        return None
    m = mask_feat.reshape(1, 1, mask_feat.shape[-2], mask_feat.shape[-1])
    m = F.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)
    return m[0, 0].cpu().numpy()  # (H,W) in [0,1]


def overlay(img_uint8, mask01, alpha=0.6):
    """Blend a red tint over the image where the mask is high. img (H,W,3) uint8, mask (H,W)."""
    red = np.zeros_like(img_uint8, dtype=np.float32)
    red[..., 0] = 255.0
    m = (mask01[..., None] * alpha).astype(np.float32)
    out = (1.0 - m) * img_uint8.astype(np.float32) + m * red
    return out.clip(0, 255).astype(np.uint8)


def to_uint8_hwc(raw, image_key):
    img = raw
    if img.dtype == np.uint8:
        return img
    img = np.asarray(img, dtype=np.float32)
    if img.max() <= 1.5:  # model-space [-0.5,0.5] or [0,1]
        img = (img + 0.5) * 255.0 if img.min() < 0 else img * 255.0
    return img.clip(0, 255).astype(np.uint8)


def main():
    args, conf = build_conf()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model + checkpoint
    model = Dreamer(conf)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint {args.checkpoint} (epoch {ckpt.get("epoch", "?")})')
    if model.wm.encoder.encoder_image.attn is None:
        raise SystemExit('Encoder has no spatial-attention head. Re-run with --use_spatial_attention True.')

    # Env + preprocessing (mirror generator.create_policy)
    env = create_env(conf.env_id, conf.env_no_terminal, conf.env_time_limit,
                     conf.env_action_repeat, worker_id=0, conf=conf)
    preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                              image_key=conf.image_key,
                              map_categorical=conf.map_channels if conf.map_categorical else None,
                              map_key=conf.map_key,
                              action_dim=env.action_size,
                              clip_rewards=conf.clip_rewards)

    reals, overlays = [], []
    obs = env.reset()
    for t in range(args.num_steps):
        raw = to_uint8_hwc(obs[conf.image_key], conf.image_key)  # (H,W,3) uint8
        H, W = raw.shape[:2]

        batch = preprocess.apply(obs, expandTB=True)
        obs_model = map_structure(batch, torch.from_numpy)  # type: ignore
        mask01 = encoder_mask(model, obs_model, H, W)

        reals.append(raw)
        overlays.append(overlay(raw, mask01, args.alpha))

        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    image = np.stack(reals)[None]      # (1,T,H,W,3)
    image_rec = np.stack(overlays)[None]
    np.savez(args.out, image=image.astype(np.uint8), image_rec=image_rec.astype(np.uint8))
    print(f'Saved {args.num_steps} frames -> {args.out}')
    print('Render with:  python visualize.py %s   (columns: real | reconstruct=obs+mask overlay)' % args.out)


if __name__ == '__main__':
    main()
