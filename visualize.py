"""
Usage:
    python visualize.py <npz_path> [--out output.mp4] [--batch 0] [--fps 10]

Example:
    python visualize.py mlruns/0/<run_id>/artifacts/d2_wm_closed/0000001.npz
    python visualize.py mlruns/0/<run_id>/artifacts/d2_wm_dream/0000001.npz
"""
import argparse
import numpy as np
import imageio


def make_video(npz_path, out_path, batch_idx=0, fps=10):
    data = np.load(npz_path)

    image = data['image'][batch_idx]        # (T, H, W, 3)  real frames
    image_masked = data.get('image_masked') # masked input image
    image_rec = data.get('image_rec')       # reconstruction (closed-loop only)
    image_pred = data.get('image_pred')     # open-loop prediction

    frames = []
    T = image.shape[0]
    for t in range(T):
        cols = [image[t]]
        if image_masked is not None:
            cols.append(image_masked[batch_idx, t])
        if image_rec is not None:
            cols.append(image_rec[batch_idx, t])
        if image_pred is not None:
            cols.append(image_pred[batch_idx, t])

        # Add separator lines between columns
        sep = np.zeros((64, 2, 3), dtype=np.uint8)
        row = cols[0]
        for col in cols[1:]:
            row = np.concatenate([row, sep, col], axis=1)

        frames.append(row)

    # Add header label
    labels = ['real']
    if image_masked is not None:
        labels.append('masked')
    if image_rec is not None:
        labels.append('reconstruct')
    if image_pred is not None:
        labels.append('predict')
    print(f'Columns: {" | ".join(labels)}')
    print(f'Saving {T} frames → {out_path}')

    imageio.mimwrite(out_path, frames, fps=fps, quality=8)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='Path to .npz file')
    parser.add_argument('--out', default='output.mp4')
    parser.add_argument('--batch', type=int, default=0, help='Which batch item to visualize')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    make_video(args.npz, args.out, args.batch, args.fps)
