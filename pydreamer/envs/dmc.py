import os

import gym
import gym.spaces
import numpy as np


class DMC(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, gt_mask=False):
        # os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')
        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()
        else:
            from dm_control import suite
            self._env = suite.load(domain, task)
        self._action_repeat = action_repeat
        self._size = size
        self._gt_mask = gt_mask
        if camera is None:
            camera = dict(
                quadruped_walk=2,
                quadruped_run=2,
                quadruped_escape=2,
                quadruped_fetch=2,
                locom_rodent_maze_forage=1,
                locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera
        self._ignored_keys = ['pixels']  # pixels added by distracting_control wrapper, not needed
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
            else:
                raise NotImplementedError(value.dtype)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        if self._gt_mask:
            spaces['mask_gt'] = gym.spaces.Box(
                0, 1, self._size + (1,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return action

    def step(self, action):
        assert np.isfinite(action).all(), action  # type: ignore
        reward = 0
        time_step = None
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        assert time_step is not None
        obs = self.observation(time_step)
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self.observation(time_step)
        return obs

    def observation(self, time_step):
        obs = dict(time_step.observation)
        obs = {k: v for k, v in obs.items() if k not in self._ignored_keys}
        obs['image'] = self.render()
        if self._gt_mask:
            obs['mask_gt'] = self.render_mask()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)  # type: ignore

    def render_mask(self):
        """Ground-truth foreground mask from the simulator's segmentation renderer:
        1 wherever a geom (agent, ground plane, ...) is visible, 0 on background
        (skybox) pixels. Background distraction (video sky texture) doesn't add
        geoms, so this is exact regardless of distraction difficulty."""
        seg = self._env.physics.render(*self._size, camera_id=self._camera, segmentation=True)  # type: ignore
        return (seg[:, :, 0] != -1).astype(np.uint8)[:, :, None]  # (H,W,1)


class DistractingDMC(DMC):

    def __init__(self, name, action_repeat=1, size=(64, 64),
                 difficulty='easy', dynamic=False,
                 background_dataset_path='/home/csy24/DAVIS/JPEGImages/480p/',
                 custom_kwargs=None, gt_mask=False):
        from distracting_control import suite as distracting_suite
        from distracting_control import suite_utils
        domain, task = name.split('_', 1)
        if domain == 'cup':
            domain = 'ball_in_cup'

        if difficulty == 'custom':
            # Fine-grained control: bypass the difficulty preset and build the three
            # distraction kwargs ourselves from config (see config/distract.yaml).
            ck = custom_kwargs or {}
            scale = ck.get('scale', 0.1)
            num_videos = ck.get('num_videos', 4) or None  # 0/None => use all videos
            background_kwargs = camera_kwargs = color_kwargs = None
            if ck.get('background', True):
                background_kwargs = suite_utils.get_background_kwargs(
                    domain, num_videos, dynamic,
                    dataset_path=background_dataset_path,
                    video_alpha=ck.get('video_alpha', 1.0))
                background_kwargs['ground_plane_alpha'] = ck.get('ground_alpha', 1.0)
            if ck.get('camera', True):
                camera_kwargs = suite_utils.get_camera_kwargs(domain, scale, dynamic)
            if ck.get('color', True):
                color_kwargs = suite_utils.get_color_kwargs(scale, dynamic)
            self._env = distracting_suite.load(
                domain_name=domain,
                task_name=task,
                difficulty=None,  # disable preset; use explicit kwargs below
                dynamic=dynamic,
                background_dataset_path=background_dataset_path,
                background_kwargs=background_kwargs,
                camera_kwargs=camera_kwargs,
                color_kwargs=color_kwargs,
                pixels_only=False,
            )
        else:
            self._env = distracting_suite.load(
                domain_name=domain,
                task_name=task,
                difficulty=difficulty,
                dynamic=dynamic,
                background_dataset_path=background_dataset_path,
                pixels_only=False,
            )
        self._action_repeat = action_repeat
        self._size = size
        self._gt_mask = gt_mask
        self._camera = dict(
            quadruped_walk=2,
            quadruped_run=2,
            quadruped_escape=2,
            quadruped_fetch=2,
        ).get(name, 0)
        self._ignored_keys = ['pixels']  # distracting_control adds raw 240x320 pixels, use our own render
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)


class PairedDistractingDMC(DistractingDMC):
    """Single distracting env, rendered twice per frame with two different
    backgrounds. Every observation includes 'image' (background A) and 'image2'
    (background B).

    The pair is pixel-identical everywhere EXCEPT the background by construction:
    both images are rendered from the SAME physics state (same qpos → same
    foreground geometry, same cam_xpos → same camera, same mat_rgba → same
    colors); only the sky background is swapped between the two renders. So
    camera/color distractions can stay ON — they are shared by both views — and
    bg_diff = |image - image2| isolates the background alone.

    The second background is produced by reusing the background wrapper's own
    `_reset_background()` / `_apply()` (no direct texture-buffer access), so this
    works regardless of the installed distracting_control version.

    Used by pretrain_masker.py to generate clean paired augmentation data with no
    second physics instance, no state syncing, and no foreground contamination.
    """

    def __init__(self, name, action_repeat=1, size=(64, 64),
                 difficulty='easy', dynamic=False,
                 background_dataset_path='/home/csy24/DAVIS/JPEGImages/480p/',
                 custom_kwargs=None, gt_mask=False):
        super().__init__(name, action_repeat, size, difficulty, dynamic,
                         background_dataset_path, custom_kwargs, gt_mask)
        # The background wrapper owns the video pool and the apply/upload logic.
        self._bg = self._find_bg_wrapper(self._env)
        assert self._bg is not None and self._bg._video_paths, \
            'PairedDistractingDMC needs background distraction enabled with videos'

    @staticmethod
    def _find_bg_wrapper(env):
        from distracting_control import background as dcbg
        while env is not None:
            if isinstance(env, dcbg.DistractingBackgroundEnv):
                return env
            env = getattr(env, '_env', None)
        return None

    def _render_other_bg(self):
        """Render the current physics state with a different (background B) sky,
        then restore background A. Foreground/camera/color are untouched, so the
        result differs from obs['image'] only in the background."""
        physics = self._env.physics
        # Snapshot the wrapper's current background (A) state.
        saved_bg = self._bg._background
        saved_idx = self._bg._current_img_index
        # Sample and apply a fresh background (B) using the wrapper's own code.
        self._bg._reset_background()
        img = physics.render(*self._size, camera_id=self._camera)
        # Restore background A (re-apply via the wrapper's own upload path).
        self._bg._background = saved_bg
        self._bg._current_img_index = saved_idx
        self._bg._apply()
        return img

    def reset(self):
        obs = super().reset()                           # obs['image'] rendered with bg A
        obs['image2'] = self._render_other_bg()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)  # obs['image'] rendered with bg A
        obs['image2'] = self._render_other_bg()
        return obs, reward, done, info
