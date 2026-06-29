import os

import gym
import gym.spaces
import numpy as np


class DMC(gym.Env):

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
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
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)  # type: ignore


class DistractingDMC(DMC):

    def __init__(self, name, action_repeat=1, size=(64, 64),
                 difficulty='easy', dynamic=False,
                 background_dataset_path='/home/csy24/DAVIS/JPEGImages/480p/',
                 custom_kwargs=None):
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
    colors); only the sky texture (SKY_TEXTURE_INDEX) is swapped between the two
    renders. So camera/color distractions can stay ON — they are shared by both
    views — and bg_diff = |image - image2| isolates the background alone.

    Used by pretrain_masker.py to generate clean paired augmentation data with no
    second physics instance, no state syncing, and no foreground contamination.
    """

    def __init__(self, name, action_repeat=1, size=(64, 64),
                 difficulty='easy', dynamic=False,
                 background_dataset_path='/home/csy24/DAVIS/JPEGImages/480p/',
                 custom_kwargs=None):
        super().__init__(name, action_repeat, size, difficulty, dynamic,
                         background_dataset_path, custom_kwargs)
        # The background wrapper owns the video pool and the sky texture.
        self._bg = self._find_bg_wrapper(self._env)
        assert self._bg is not None and self._bg._video_paths, \
            'PairedDistractingDMC needs background distraction enabled with videos'
        self._video_alpha = (custom_kwargs or {}).get('video_alpha', 1.0)
        self._rng2 = np.random.RandomState()   # independent sampler for the 2nd bg
        self._tex_b = None

    @staticmethod
    def _find_bg_wrapper(env):
        from distracting_control import background as dcbg
        while env is not None:
            if isinstance(env, dcbg.DistractingBackgroundEnv):
                return env
            env = getattr(env, '_env', None)
        return None

    def _sky_slice(self):
        from distracting_control.background import SKY_TEXTURE_INDEX as I
        m = self._env.physics.model
        return int(m.tex_adr[I]), int(m.tex_height[I] * m.tex_width[I] * 3)

    def _build_second_bg(self):
        """Prepare a second sky texture from a different random DAVIS frame."""
        import os
        from pathlib import Path
        from distracting_control import background as dcbg
        I = dcbg.SKY_TEXTURE_INDEX
        m = self._env.physics.model
        adr, size = self._sky_slice()
        h, w = int(m.tex_height[I]), int(m.tex_width[I])
        base = dcbg.get_texture_data(m)[adr:adr + size].astype(np.float32)  # only used if alpha<1
        video = self._rng2.choice(self._bg._video_paths)
        frames = [p.name for p in Path(video).glob('*/')]
        img = dcbg.imread(os.path.join(video, self._rng2.choice(frames)))
        flat = dcbg.size_and_flatten(img, h, w)
        self._tex_b = dcbg.blend_to_background(self._video_alpha, flat, base)

    def _render_swapped(self, tex):
        """Swap in `tex` as the sky texture, render, then restore the original."""
        from distracting_control import background as dcbg
        from dm_control.mujoco.wrapper import mjbindings
        physics = self._env.physics
        adr, size = self._sky_slice()
        data = dcbg.get_texture_data(physics.model)
        saved = data[adr:adr + size].copy()            # current background (A)

        def _upload():
            with physics.contexts.gl.make_current() as ctx:
                ctx.call(mjbindings.mjlib.mjr_uploadTexture,
                         physics.model.ptr, physics.contexts.mujoco.ptr,
                         dcbg.SKY_TEXTURE_INDEX)

        data[adr:adr + size] = tex
        _upload()
        img = physics.render(*self._size, camera_id=self._camera)
        data[adr:adr + size] = saved                   # restore A (required in static mode)
        _upload()
        return img

    def reset(self):
        obs = super().reset()                          # obs['image'] rendered with bg A
        self._build_second_bg()                        # sample bg B for this episode
        obs['image2'] = self._render_swapped(self._tex_b)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)  # obs['image'] rendered with bg A
        obs['image2'] = self._render_swapped(self._tex_b)
        return obs, reward, done, info
