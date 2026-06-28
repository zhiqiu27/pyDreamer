"""
Pixel-level Inverse Dynamics Model (IDM) test.

Verifies that consecutive image pairs (o_t, o_{t+1}) can predict the
action a_{t+1} executed between them, using DMC cartpole episode data.

Convention (matches pydreamer/data.py):
    action[t] led to obs[t], so the action between obs[t] and obs[t+1]
    is action[t+1]. Pairs: (image[:-1], image[1:]) -> action[1:].
    Pairs crossing episode boundaries (reset[t+1] == True) are dropped.

Usage:
    python test/test_idm.py --data_dir /path/to/mlruns/0/RUN_ID/artifacts/episodes/0

Output:
    test/results/idm_loss.png   -- train/val MSE loss curve
    test/results/idm_pred.png   -- predicted vs ground-truth actions (val set)
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Optional wandb — imported lazily so the script works without it
wandb = None

# ── Model ─────────────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    4-layer CNN matching pydreamer ConvEncoder (encoders.py).
    Input:  (N, C, H, W)  float in [-0.5, 0.5]
    Output: (N, depth*32)
    """
    def __init__(self, in_channels=3, depth=32):
        super().__init__()
        d = depth
        self.out_dim = d * 32
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, d,     4, 2), nn.ELU(),
            nn.Conv2d(d,           d * 2, 4, 2), nn.ELU(),
            nn.Conv2d(d * 2,       d * 4, 4, 2), nn.ELU(),
            nn.Conv2d(d * 4,       d * 8, 4, 2), nn.ELU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class IDM(nn.Module):
    """
    Two-stream inverse dynamics model.
    Each frame is encoded independently (shared weights), embeddings are
    concatenated, and an MLP predicts the action.
    """
    def __init__(self, action_dim=1, cnn_depth=32, hidden_dim=256):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=3, depth=cnn_depth)
        embed_dim = self.encoder.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs_t, obs_tp1):
        h_t   = self.encoder(obs_t)
        h_tp1 = self.encoder(obs_tp1)
        return self.mlp(torch.cat([h_t, h_tp1], dim=-1))


# ── Dataset ───────────────────────────────────────────────────────────────────

class IDMDataset(Dataset):
    """
    Loads (obs_t, obs_{t+1}, action_{t+1}) pairs from episode .npz files.
    """
    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, 'ep*.npz')))
        if not files:
            # Try one level deeper (mlflow sometimes nests under worker id)
            files = sorted(glob.glob(os.path.join(data_dir, '*', 'ep*.npz')))
        assert files, f'No ep*.npz files found under {data_dir}'
        print(f'Found {len(files)} episode files in {data_dir}')

        obs_t_list   = []
        obs_tp1_list = []
        action_list  = []
        skipped = 0

        for path in files:
            data = np.load(path, allow_pickle=False)

            # Image: stored as (T, H, W, C) uint8, sometimes as image_t: (H, W, C, T)
            if 'image' in data:
                image = data['image']               # (T, H, W, C)
            elif 'image_t' in data:
                image = data['image_t'].transpose(3, 0, 1, 2)  # HWCT -> THWC
            else:
                skipped += 1
                continue

            action = data['action'].astype(np.float32)  # (T, A) continuous

            T = len(image)
            if T < 2:
                skipped += 1
                continue

            if 'reset' in data:
                reset = data['reset'].astype(bool)
            else:
                reset = np.zeros(T, dtype=bool)
            reset[0] = True  # episode always starts with reset

            # action[t] led to obs[t]; action between obs[t] and obs[t+1] is action[t+1]
            # Drop pairs where obs[t+1] starts a new episode
            valid = ~reset[1:]                        # (T-1,)

            obs_t_list.append(image[:-1][valid])      # (N, H, W, C) uint8
            obs_tp1_list.append(image[1:][valid])
            action_list.append(action[1:][valid])     # (N, A)

        assert obs_t_list, 'All files were skipped or had no valid pairs'
        if skipped:
            print(f'  Skipped {skipped} files (missing image or too short)')

        self.obs_t   = np.concatenate(obs_t_list,   axis=0)  # (N, H, W, C)
        self.obs_tp1 = np.concatenate(obs_tp1_list, axis=0)
        self.action  = np.concatenate(action_list,  axis=0)  # (N, A)

        print(f'  Total pairs: {len(self):,}')
        print(f'  obs shape: {self.obs_t.shape}  action shape: {self.action.shape}')
        print(f'  action range: [{self.action.min():.3f}, {self.action.max():.3f}]')

    def __len__(self):
        return len(self.obs_t)

    def __getitem__(self, idx):
        # uint8 HWC  →  float CHW in [-0.5, 0.5]
        obs_t   = torch.from_numpy(self.obs_t[idx]).float().div(255.0).sub(0.5).permute(2, 0, 1)
        obs_tp1 = torch.from_numpy(self.obs_tp1[idx]).float().div(255.0).sub(0.5).permute(2, 0, 1)
        action  = torch.from_numpy(self.action[idx])
        return obs_t, obs_tp1, action


# ── Training ──────────────────────────────────────────────────────────────────

def random_baseline_mse(action_np):
    """MSE when predicting the training-set mean for every sample (dumb baseline)."""
    mean = action_np.mean(axis=0, keepdims=True)
    return float(np.mean((action_np - mean) ** 2))


def init_wandb(args):
    global wandb
    if not args.wandb:
        return
    try:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )
    except ImportError:
        print('wandb not installed, skipping. pip install wandb to enable.')
        wandb = None


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    init_wandb(args)
    device = torch.device(args.device)

    # ── Data ──
    dataset = IDMDataset(args.data_dir)

    n_val   = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Random baseline: predict training mean
    train_actions = dataset.action[[i for i in train_ds.indices]]
    baseline_mse  = random_baseline_mse(train_actions)
    print(f'\nRandom baseline MSE (predict mean): {baseline_mse:.4f}')

    # ── Model ──
    action_dim = dataset.action.shape[-1]
    model = IDM(action_dim=action_dim, cnn_depth=args.cnn_depth,
                hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'IDM parameters: {n_params:,}\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Loop ──
    train_losses = []
    val_losses   = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running = []
        for obs_t, obs_tp1, action in train_loader:
            obs_t   = obs_t.to(device)
            obs_tp1 = obs_tp1.to(device)
            action  = action.to(device)
            pred    = model(obs_t, obs_tp1)
            loss    = F.mse_loss(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running.append(loss.item())

        train_mse = float(np.mean(running))
        train_losses.append(train_mse)

        # Validate
        model.eval()
        running = []
        preds_all  = []
        labels_all = []
        with torch.no_grad():
            for obs_t, obs_tp1, action in val_loader:
                obs_t   = obs_t.to(device)
                obs_tp1 = obs_tp1.to(device)
                action  = action.to(device)
                pred    = model(obs_t, obs_tp1)
                loss    = F.mse_loss(pred, action)
                running.append(loss.item())
                preds_all.append(pred.cpu())
                labels_all.append(action.cpu())

        val_mse = float(np.mean(running))
        val_losses.append(val_mse)

        preds_all  = torch.cat(preds_all).numpy()
        labels_all = torch.cat(labels_all).numpy()
        corr = float(np.corrcoef(preds_all[:, 0], labels_all[:, 0])[0, 1])

        print(f'Epoch {epoch:3d}/{args.epochs}'
              f'  train_mse={train_mse:.4f}'
              f'  val_mse={val_mse:.4f}'
              f'  corr={corr:+.3f}'
              f'  baseline={baseline_mse:.4f}')

        if wandb:
            wandb.log({
                'epoch':        epoch,
                'train/mse':    train_mse,
                'val/mse':      val_mse,
                'val/corr':     corr,
                'baseline/mse': baseline_mse,
            }, step=epoch)

    # ── Plots ──
    _plot_loss(train_losses, val_losses, baseline_mse, args)
    _plot_pred(preds_all, labels_all, args)

    if wandb:
        wandb.log({
            'charts/loss_curve': wandb.Image(os.path.join(args.out_dir, 'idm_loss.png')),
            'charts/pred_scatter': wandb.Image(os.path.join(args.out_dir, 'idm_pred.png')),
        })
        wandb.finish()


def _plot_loss(train_losses, val_losses, baseline_mse, args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    plt.figure(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='train MSE')
    plt.plot(epochs, val_losses,   label='val MSE')
    plt.axhline(baseline_mse, color='gray', linestyle='--', label='baseline (predict mean)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('IDM — Pixel Inverse Dynamics (DMC cartpole)')
    plt.legend()
    plt.tight_layout()
    out = os.path.join(args.out_dir, 'idm_loss.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'\nSaved loss curve → {out}')


def _plot_pred(preds, labels, args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Scatter of predicted vs ground-truth for action dim 0
    n = min(2000, len(preds))
    idx = np.random.choice(len(preds), n, replace=False)
    p, l = preds[idx, 0], labels[idx, 0]
    corr = float(np.corrcoef(p, l)[0, 1])

    plt.figure(figsize=(5, 5))
    plt.scatter(l, p, alpha=0.2, s=4)
    lims = [min(l.min(), p.min()), max(l.max(), p.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('Ground-truth action')
    plt.ylabel('Predicted action')
    plt.title(f'IDM predictions (val set, n={n})\ncorr={corr:.3f}')
    plt.tight_layout()
    out = os.path.join(args.out_dir, 'idm_pred.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved prediction scatter → {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Pixel IDM test on DMC cartpole episodes')
    p.add_argument('--data_dir', required=True,
                   help='Path to directory containing ep*.npz episode files, '
                        'e.g. mlruns/0/RUN_ID/artifacts/episodes/0')
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch_size',  type=int,   default=256)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--cnn_depth',   type=int,   default=32,
                   help='CNN depth multiplier (same as pydreamer cnn_depth)')
    p.add_argument('--hidden_dim',  type=int,   default=256,
                   help='MLP hidden dim')
    p.add_argument('--val_frac',    type=float, default=0.1,
                   help='Fraction of pairs held out for validation')
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',       default='test/results',
                   help='Directory to save loss curve and prediction plot')
    p.add_argument('--wandb',         action='store_true',
                   help='Log metrics to Weights & Biases')
    p.add_argument('--wandb_project', default='pydreamer-idm',
                   help='W&B project name')
    p.add_argument('--wandb_run',     default=None,
                   help='W&B run name (default: auto)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'Device: {args.device}')
    print(f'Data:   {args.data_dir}')
    print(f'Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}\n')
    train(args)
