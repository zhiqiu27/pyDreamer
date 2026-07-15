"""Visualize the GT-mask supersampling pipeline on (Distracting)DMC.

Rolls out an episode with random actions and writes a side-by-side video:

    [ raw image | premultiplied masked image | soft mask ]

All three panels are the 64x64 tensors the model actually consumes
(obs['image'], obs['image_masked_gt'], obs['mask_gt'] from
DMC._render_gt_supersampled), upscaled with nearest-neighbor so the
anti-aliased mask edges are visible.

Standalone: only dm_control / distracting_control / cv2 are required. If `gym`
is not installed a minimal stub is injected (only observation_space needs it,
and this script never calls that).

Usage (from the repo root, in the `dmc` conda env):
    python vis_gt_mask.py --task cheetah_run --steps 200 --supersample 4 \
        --davis "C:/Users/zhiqiu/Desktop/distracting_control/DAVIS/JPEGImages/480p" \
        --out gt_mask_vis.mp4
    python vis_gt_mask.py --plain   # plain DMC, no distraction background
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np


def ensure_gym_stub():
    try:
        import gym  # noqa: F401
    except ModuleNotFoundError:
        gym = types.ModuleType('gym')
        gym.Env = object
        gym.spaces = types.SimpleNamespace(Box=lambda *a, **k: None,
                                           Dict=lambda *a, **k: None)
        sys.modules['gym'] = gym
        sys.modules['gym.spaces'] = gym.spaces


def load_dmc_module():
    # Load pydreamer/envs/dmc.py directly: the package __init__ pulls in wrappers
    # and other deps (torch etc.) that aren't needed for rendering.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'pydreamer', 'envs', 'dmc.py')
    spec = importlib.util.spec_from_file_location('dmc_vis', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def upscale(img, k):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='cheetah_run')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--supersample', type=int, default=4)
    p.add_argument('--difficulty', default='easy')
    p.add_argument('--dynamic', action='store_true', default=True)
    p.add_argument('--plain', action='store_true', help='plain DMC instead of DistractingDMC')
    p.add_argument('--davis', default=r'C:\Users\zhiqiu\Desktop\distracting_control\DAVIS\JPEGImages\480p')
    p.add_argument('--out', default='gt_mask_vis.mp4')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--view-scale', type=int, default=4, help='nearest-neighbor upscale for viewing')
    args = p.parse_args()

    ensure_gym_stub()
    import cv2
    dmc = load_dmc_module()

    if args.plain:
        env = dmc.DMC(args.task, gt_mask=True, gt_mask_supersample=args.supersample)
    else:
        env = dmc.DistractingDMC(args.task, difficulty=args.difficulty, dynamic=args.dynamic,
                                 background_dataset_path=args.davis,
                                 gt_mask=True, gt_mask_supersample=args.supersample)

    spec = env._env.action_spec()

    def random_action():
        return np.random.uniform(spec.minimum, spec.maximum).astype(np.float32)

    k = args.view_scale
    header_h = 24
    labels = ['image_hr (pre-downsample)', 'image (raw)', 'image_masked_gt', 'mask_gt (soft)']
    frames = []
    hr_size = (64 * args.supersample, 64 * args.supersample)
    panel_size = (64 * k, 64 * k)

    def render_hr():
        # Same physics state as the obs render, so identical to the internal
        # high-res image _render_gt_supersampled premultiplied from.
        img = env._env.physics.render(*hr_size, camera_id=env._camera)
        if hr_size != panel_size:
            img = cv2.resize(img, panel_size, interpolation=cv2.INTER_NEAREST)
        return img

    obs = env.reset()
    frac_pixels = total_pixels = 0
    for t in range(args.steps):
        m = obs['mask_gt']
        frac_pixels += int(((m > 0) & (m < 255)).sum())
        total_pixels += m.size
        mask_rgb = np.repeat(obs['mask_gt'], 3, axis=2)
        panels = [render_hr(),
                  upscale(obs['image'], k),
                  upscale(obs['image_masked_gt'], k),
                  upscale(mask_rgb, k)]
        sep = np.full((panels[0].shape[0], 2, 3), 255, np.uint8)
        row = np.concatenate([x for pair in zip(panels, [sep] * len(panels)) for x in pair][:-1], axis=1)
        header = np.zeros((header_h, row.shape[1], 3), np.uint8)
        pw = panels[0].shape[1] + 2
        for i, text in enumerate(labels):
            cv2.putText(header, text, (i * pw + 4, header_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(np.concatenate([header, row], axis=0))

        obs, reward, done, info = env.step(random_action())
        if done:
            obs = env.reset()

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    assert writer.isOpened(), f'failed to open video writer for {args.out}'
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f'wrote {len(frames)} frames -> {os.path.abspath(args.out)} ({w}x{h} @ {args.fps}fps)')
    print(f'soft (fractional) mask pixels: {frac_pixels}/{total_pixels} '
          f'({100.0 * frac_pixels / max(total_pixels, 1):.2f}%) — should be >0 when supersample>1')


if __name__ == '__main__':
    main()
