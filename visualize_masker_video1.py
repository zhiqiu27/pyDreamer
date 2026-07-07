"""Apply a pre-trained ObservationMasker to a random-policy rollout and save
a side-by-side video: original | masked | mask grayscale | binary mask | mask histogram.

No bg_diff or image2 needed — mask_net predicts from a single image at
inference (masker.forward), just like it will inside the WorldModel.

Usage:
    python visualize_masker_video.py \
        --run_id <mlflow_run_id> \
        --env_id DistractingDMC-easy-cheetah_run \
        --steps 300 \
        --out masker_video.mp4 \
        --action_dim 6
"""
import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pydreamer.models.masker import ObservationMasker


# ── helpers ──────────────────────────────────────────────────────────────────

def load_masker_checkpoint(run_id, artifact_path, map_location):
    """Download the masker checkpoint of an MLflow run by its run id and load it."""
    import tempfile
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, artifact_path, tmpdir)
        return torch.load(path, map_location=map_location, weights_only=False)

def load_custom_kwargs(distract_config='config/distract.yaml'):
    """Read the fine-grained distraction params from config/distract.yaml,
    mirroring what create_env() builds for difficulty=custom."""
    import yaml
    with open(distract_config) as f:
        cfg = yaml.safe_load(f) or {}
    d = cfg.get('distract', cfg)  # support both nested and flat layouts
    return dict(
        scale=d.get('distract_scale', 0.1),
        num_videos=d.get('distract_num_videos', 4),
        video_alpha=d.get('distract_video_alpha', 1.0),
        ground_alpha=d.get('distract_ground_alpha', 1.0),
        background=d.get('distract_background', True),
        camera=d.get('distract_camera', True),
        color=d.get('distract_color', True),
    )


def make_env(env_id, action_repeat=2, dynamic=True,
             bg_path='/home/csy24/DAVIS/JPEGImages/480p/',
             distract_config='config/distract.yaml'):
    """Create a DistractingDMC (or plain DMC) environment.

    dynamic=True: background video advances each step (moving background).
    dynamic=False: background is fixed per episode (static).

    difficulty=custom (env_id like DistractingDMC-custom-static-cheetah_run)
    reads fine-grained params from config/distract.yaml, matching create_env().
    A static/dynamic token in env_id overrides the `dynamic` argument.
    """
    from pydreamer.envs.dmc import DistractingDMC, DMC
    if env_id.startswith('DistractingDMC-'):
        parts = env_id.split('-', maxsplit=3)
        difficulty = 'easy'
        name = parts[-1].lower()
        for tok in parts[1:-1]:
            if tok in ('easy', 'medium', 'hard', 'custom'):
                difficulty = tok
            elif tok == 'dynamic':
                dynamic = True
            elif tok == 'static':
                dynamic = False
        custom_kwargs = load_custom_kwargs(distract_config) if difficulty == 'custom' else None
        return DistractingDMC(name, action_repeat=action_repeat,
                              difficulty=difficulty, dynamic=dynamic,
                              background_dataset_path=bg_path,
                              custom_kwargs=custom_kwargs)
    elif env_id.startswith('DMC-'):
        return DMC(env_id.split('-', 1)[1].lower(), action_repeat=action_repeat)
    else:
        raise ValueError(f'Unsupported env_id: {env_id}')


def preprocess_frame(img_uint8, device):
    """uint8 (H,W,C) -> float tensor (1,1,C,H,W) in [-0.5, 0.5]."""
    x = img_uint8.astype(np.float32) / 255.0 - 0.5
    x = x.transpose(2, 0, 1)[None, None]           # (1,1,C,H,W)
    return torch.from_numpy(x).to(device)


def to_display(tensor_chw):
    """float (C,H,W) in [-0.5,0.5] -> uint8 (H,W,C) in [0,255]."""
    arr = tensor_chw.cpu().numpy().transpose(1, 2, 0)
    return (np.clip(arr + 0.5, 0, 1) * 255).astype(np.uint8)


def mask_gray_to_display(mask_hw):
    """float mask (H,W) in [0,1] -> RGB grayscale, 0 black and 1 white."""
    gray = (np.clip(mask_hw, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def mask_binary_to_display(mask_hw, threshold=0.5):
    """Threshold mask to pure black/white RGB for quick keep/drop inspection."""
    binary = (mask_hw >= threshold).astype(np.uint8) * 255
    return np.repeat(binary[..., None], 3, axis=-1)


def mask_hist_to_display(mask_hw, height, width, bins=20):
    """Render a compact per-frame histogram for mask values in [0,1]."""
    hist, _ = np.histogram(mask_hw, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    if hist.max() > 0:
        hist /= hist.max()

    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 6, 4, 5, 8
    plot_h = max(1, height - margin_t - margin_b)
    plot_w = max(1, width - margin_l - margin_r)
    baseline = margin_t + plot_h

    # Axes.
    panel[margin_t:baseline + 1, margin_l] = 0
    panel[baseline, margin_l:margin_l + plot_w] = 0

    bar_w = max(1, plot_w // bins)
    for i, v in enumerate(hist):
        x0 = margin_l + i * plot_w // bins + 1
        x1 = min(margin_l + (i + 1) * plot_w // bins, margin_l + plot_w)
        if x1 <= x0:
            x1 = min(x0 + bar_w, margin_l + plot_w)
        bar_h = int(v * (plot_h - 1))
        y0 = baseline - bar_h
        panel[y0:baseline, x0:x1] = 40

    # Mark low/high ends so the horizontal axis is easy to read in video.
    panel[baseline + 1:height, margin_l:margin_l + 2] = 0
    panel[baseline + 1:height, margin_l + plot_w - 2:margin_l + plot_w] = 0
    return panel


def save_video(frames, path, fps=20):
    """Save list of (H,W,3) uint8 arrays as mp4 via imageio."""
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        return True
    except Exception as e:
        print(f'[imageio] {e}')
        return False


def save_frames(frames, path):
    """Fall-back: save a montage PNG (one row per frame sampled evenly)."""
    import math
    n = min(12, len(frames))
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    cols = 4
    rows = math.ceil(n / cols)
    H, W = frames[0].shape[:2]
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = frames[i]
    plt.imsave(path, canvas)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_id', required=True,
                   help='MLflow run id; the masker checkpoint is loaded from its artifacts')
    p.add_argument('--artifact_path', default='checkpoints/masker.pt',
                   help='artifact path of the masker checkpoint within the run')
    p.add_argument('--env_id', default='DistractingDMC-easy-cheetah_run')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--action_dim', type=int, default=6)
    p.add_argument('--masker_depth', type=int, default=32)
    p.add_argument('--actor_dist', default='tanh_normal')
    p.add_argument('--action_repeat', type=int, default=2)
    p.add_argument('--dynamic', type=lambda x: x.lower() != 'false', default=True,
                   help='dynamic background (default True — background moves each step)')
    p.add_argument('--bg_path', default='/home/csy24/DAVIS/JPEGImages/480p/')
    p.add_argument('--distract_config', default='config/distract.yaml',
                   help='config with fine-grained distraction params (used when env_id difficulty=custom)')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--out', default='masker_video.mp4')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = torch.device(args.device)

    # ── load masker ──────────────────────────────────────────────────────────
    ckpt = load_masker_checkpoint(args.run_id, args.artifact_path, map_location=device)
    masker = ObservationMasker(
        in_channels=3, action_dim=args.action_dim,
        depth=args.masker_depth, actor_dist=args.actor_dist,
    ).to(device)
    masker.load_state_dict(ckpt['masker_state_dict'])
    masker.eval()
    step_loaded = ckpt.get('epoch', '?')
    print(f'Loaded masker (step {step_loaded}) from run {args.run_id} ({args.artifact_path})')

    # ── create env ───────────────────────────────────────────────────────────
    print(f'Creating env: {args.env_id}  dynamic={args.dynamic}')
    env = make_env(args.env_id, action_repeat=args.action_repeat,
                   dynamic=args.dynamic, bg_path=args.bg_path,
                   distract_config=args.distract_config)
    action_space = env.action_space

    # ── rollout ──────────────────────────────────────────────────────────────
    obs = env.reset()
    composite_frames = []   # side-by-side (H, 5*W, 3) uint8 frames

    print(f'Running {args.steps} steps with random policy ...')
    for step in range(args.steps):
        img_u8 = obs['image']                           # (H, W, C) uint8

        # Masker inference — single image, no image2 needed
        with torch.no_grad():
            x = preprocess_frame(img_u8, device)        # (1,1,C,H,W)
            masked, mask = masker(x)                    # (1,1,C,H,W), (1,1,1,H,W)

        orig_rgb  = img_u8                              # (H, W, 3) uint8
        m         = mask[0, 0, 0].cpu().numpy()        # (H, W) in [0, 1]
        # Zero out the background: mask≈0 -> black, mask≈1 -> original pixel.
        masked_rgb = (orig_rgb.astype(np.float32) * m[..., None]).astype(np.uint8)
        mask_gray = mask_gray_to_display(m)
        mask_binary = mask_binary_to_display(m)
        mask_hist = mask_hist_to_display(m, *orig_rgb.shape[:2])

        side_by_side = np.concatenate(
            [orig_rgb, masked_rgb, mask_gray, mask_binary, mask_hist],
            axis=1,
        )  # (H, 5W, 3)
        composite_frames.append(side_by_side)

        action = action_space.sample()
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

        if (step + 1) % 50 == 0:
            print(f'  step {step+1}/{args.steps}  mask_mean={mask.mean():.3f}')

    print(f'Collected {len(composite_frames)} frames, saving ...')

    # ── save ──────────────────────────────────────────────────────────────────
    ext = os.path.splitext(args.out)[-1].lower()
    if ext in ('.mp4', '.gif', '.avi'):
        ok = save_video(composite_frames, args.out, fps=args.fps)
        if not ok:
            fallback = args.out.replace(ext, '_montage.png')
            save_frames(composite_frames, fallback)
            print(f'imageio failed — saved montage to {fallback}')
        else:
            print(f'Saved video: {args.out}  ({len(composite_frames)} frames @ {args.fps} fps)')
    else:
        save_frames(composite_frames, args.out)
        print(f'Saved montage: {args.out}')

    print('Done.')


if __name__ == '__main__':
    main()
