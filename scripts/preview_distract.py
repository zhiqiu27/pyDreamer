"""Render cartpole observations under different DistractingDMC settings into one
labeled comparison image, to visualize what each distract param does.

Run (on a machine with working MuJoCo GL):
    MUJOCO_GL=egl python scripts/preview_distract.py
Output: distract_preview.png in repo root.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dm_control import suite as clean_suite
from distracting_control import suite as dsuite
from distracting_control import suite_utils as su

PATH = '/home/csy24/DAVIS/JPEGImages/480p/'
DOMAIN, TASK = 'cartpole', 'balance'
SEED = 1          # fixed -> same cart pose across all tiles
WARMUP = 8        # steps so the scene is non-trivial


def render(env):
    return env.physics.render(64, 64, camera_id=0)


def rollout(env):
    env.reset()
    a = np.zeros(env.action_spec().shape)
    for _ in range(WARMUP):
        env.step(a)
    return render(env)


def custom(scale=0.1, num_videos=4, video_alpha=1.0, ground_alpha=0.3,
           bg=True, cam=True, col=True, dynamic=False):
    bg_k = cam_k = col_k = None
    if bg:
        bg_k = su.get_background_kwargs(DOMAIN, num_videos or None, dynamic,
                                        dataset_path=PATH, video_alpha=video_alpha)
        bg_k['ground_plane_alpha'] = ground_alpha
    if cam:
        cam_k = su.get_camera_kwargs(DOMAIN, scale, dynamic)
    if col:
        col_k = su.get_color_kwargs(scale, dynamic)
    return dsuite.load(DOMAIN, TASK, difficulty=None, dynamic=dynamic,
                       background_dataset_path=PATH,
                       background_kwargs=bg_k, camera_kwargs=cam_k, color_kwargs=col_k,
                       task_kwargs={'random': SEED}, pixels_only=False)


tiles = []
# Clean reference (no distraction at all)
tiles.append(('clean (no distract)',
              rollout(clean_suite.load(DOMAIN, TASK, task_kwargs={'random': SEED}))))
# Isolate each distraction type
tiles.append(('background only (alpha=1.0)', rollout(custom(bg=True, cam=False, col=False))))
tiles.append(('background alpha=0.5',        rollout(custom(bg=True, cam=False, col=False, video_alpha=0.5))))
tiles.append(('camera only scale=0.3',       rollout(custom(bg=False, cam=True, col=False, scale=0.3))))
tiles.append(('color only scale=0.3',        rollout(custom(bg=False, cam=False, col=True, scale=0.3))))
# Combined presets
tiles.append(('all easy (scale=0.1)',        rollout(custom(scale=0.1, num_videos=4))))
tiles.append(('all medium (scale=0.2)',      rollout(custom(scale=0.2, num_videos=8))))
tiles.append(('all hard (scale=0.3, all)',   rollout(custom(scale=0.3, num_videos=0))))

n = len(tiles)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
for ax, (title, img) in zip(axes.flat, tiles):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
for ax in axes.flat[n:]:
    ax.axis('off')
fig.tight_layout()
out = 'distract_preview.png'
fig.savefig(out, dpi=120)
print('saved', os.path.abspath(out))
