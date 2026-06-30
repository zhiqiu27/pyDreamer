"""Visualize a pre-trained ObservationMasker.

Loads a masker checkpoint (masker.pt) and a few frames from the generator's
npz episodes, then renders a grid so you can SEE what the mask learned:

    image | image2 | bg_diff | fg_target | mask | masked

  - image / image2 : the two background-augmented views (same foreground)
  - bg_diff        : |image - image2|, the localizing signal (bright = background)
  - fg_target      : exp(-bg_diff/temp), the supervision target (bright = foreground)
  - mask           : what mask_net predicts from a SINGLE image (bright = keep)
  - masked         : image * mask (what the encoder/decoder will actually see)

A good mask looks like fg_target: bright on the agent, dark on the background.

Usage (on the Linux box where the run lives):
    python visualize_masker.py \
        --checkpoint <mlruns>/.../artifacts/checkpoints/masker.pt \
        --data_dir   <mlruns>/.../artifacts/episodes \
        --out        masker_viz.png \
        --action_dim 6 --n 6 --mask_temp 0.5
"""
import argparse
import glob
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pydreamer.models.masker import ObservationMasker


def find_npz(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
    if not files:
        raise FileNotFoundError(f'No .npz episodes under {data_dir}')
    return files


def load_pair(files, n):
    """Return n frames of (image, image2) as uint8 (n, H, W, C), picking the
    first episode that actually has image2.

    Generator saves images transposed as (H,W,C,T) under key 'image_t'/'image2_t'
    for better npz compression. We transpose back to (T,H,W,C).
    """
    for f in files:
        d = np.load(f)
        # keys are 'image_t' / 'image2_t' (HWCT) after generator's transpose
        if 'image2_t' not in d.files:
            continue
        img  = d['image_t'].transpose(3, 0, 1, 2)    # (T, H, W, C) uint8
        img2 = d['image2_t'].transpose(3, 0, 1, 2)
        T = img.shape[0]
        idx = np.linspace(0, T - 1, num=min(n, T)).astype(int)
        return img[idx], img2[idx], os.path.basename(f)
    raise RuntimeError(
        'No episode with image2_t found.\n'
        'Keys in first file: ' + str(list(np.load(files[0]).files))
        if files else 'No .npz files found at all.'
    )


def to_model(x_uint8, device):
    # match preprocessing.to_image: uint8 (n,H,W,C) -> float (1,n,C,H,W) in [-0.5,0.5]
    x = x_uint8.astype(np.float32) / 255.0 - 0.5
    x = x.transpose(0, 3, 1, 2)[None]                  # (1, n, C, H, W)
    return torch.from_numpy(x).to(device)


def show(ax, arr, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out', default='masker_viz.png')
    p.add_argument('--n', type=int, default=6, help='number of frames')
    p.add_argument('--action_dim', type=int, default=6)
    p.add_argument('--masker_depth', type=int, default=32)
    p.add_argument('--mask_temp', type=float, default=0.5)
    p.add_argument('--actor_dist', default='tanh_normal')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = torch.device(args.device)

    # --- model ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    masker = ObservationMasker(in_channels=3, action_dim=args.action_dim,
                               depth=args.masker_depth, actor_dist=args.actor_dist).to(device)
    masker.load_state_dict(ckpt['masker_state_dict'])
    masker.eval()
    print(f'Loaded checkpoint (step {ckpt.get("epoch", "?")}) from {args.checkpoint}')

    # --- data ---
    files = find_npz(args.data_dir)
    img_u8, img2_u8, fname = load_pair(files, args.n)
    print(f'Using {img_u8.shape[0]} frames from {fname}')

    v1 = to_model(img_u8, device)                       # (1, n, C, H, W)
    v2 = to_model(img2_u8, device)

    with torch.no_grad():
        masked, mask = masker(v1)                       # (1,n,C,H,W), (1,n,1,H,W)
        bg_diff = (v1 - v2).abs().mean(dim=2, keepdim=True)
        bg_diff = bg_diff / (bg_diff.mean() + 1e-6)
        fg_target = torch.exp(-bg_diff / args.mask_temp)

    # to numpy, drop batch dim, (n,...)
    img   = (v1[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5).clip(0, 1)
    img2  = (v2[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5).clip(0, 1)
    mskd  = (masked[0].cpu().numpy().transpose(0, 2, 3, 1) + 0.5).clip(0, 1)
    msk   = mask[0, :, 0].cpu().numpy()
    bgd   = bg_diff[0, :, 0].cpu().numpy()
    fgt   = fg_target[0, :, 0].cpu().numpy()

    n = img.shape[0]
    cols = ['image', 'image2', 'bg_diff', 'fg_target', 'mask', 'masked']
    fig, axes = plt.subplots(n, len(cols), figsize=(2 * len(cols), 2 * n))
    if n == 1:
        axes = axes[None]
    for r in range(n):
        show(axes[r, 0], img[r],  cols[0] if r == 0 else '')
        show(axes[r, 1], img2[r], cols[1] if r == 0 else '')
        show(axes[r, 2], bgd[r],  cols[2] if r == 0 else '', cmap='inferno')
        show(axes[r, 3], fgt[r],  cols[3] if r == 0 else '', cmap='viridis', vmin=0, vmax=1)
        show(axes[r, 4], msk[r],  cols[4] if r == 0 else '', cmap='viridis', vmin=0, vmax=1)
        show(axes[r, 5], mskd[r], cols[5] if r == 0 else '')

    fig.suptitle(f'masker step {ckpt.get("epoch","?")}  |  '
                 f'mask_mean={msk.mean():.3f}  temp={args.mask_temp}', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
