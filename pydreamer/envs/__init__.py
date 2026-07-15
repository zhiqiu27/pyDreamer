# Ignore annoying warnings from imported envs
import warnings
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

import gym
import numpy as np

from .wrappers import *


def create_env(env_id: str, no_terminal: bool, env_time_limit: int, env_action_repeat: int, worker_id: int, conf=None):

    if env_id.startswith('MiniGrid-'):
        from .minigrid import MiniGrid
        env = MiniGrid(env_id)

    elif env_id.startswith('Atari-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)

    elif env_id.startswith('AtariGray-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat, grayscale=True)

    elif env_id.startswith('MiniWorld-'):
        import gym_miniworld.wrappers as wrap
        env = gym.make(env_id)
        env = wrap.DictWrapper(env)
        env = wrap.MapWrapper(env)
        env = wrap.AgentPosWrapper(env)
        if env_id.startswith('MiniWorld-ScavengerHunt'):
            env = wrap.GoalPosWrapper(env)
            env = wrap.GoalVisibleWrapper(env)
            env = wrap.GoalVisAgeWrapper(env)

    elif env_id.startswith('DmLab-'):
        from .dmlab import DmLab
        env = DmLab(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat)
        env = DictWrapper(env)
    
    elif env_id.startswith('DMM-'):
        from .dmm import DMMEnv
        env = DMMEnv(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat, worker_id=worker_id)
        env = DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from .minerl import MineRL
        constr = lambda: MineRL(env_id, action_repeat=env_action_repeat)
        env = RestartOnExceptionWrapper(constr)

    elif env_id.startswith('PairedDistractingDMC-') or env_id.startswith('DistractingDMC-'):
        from .dmc import DistractingDMC, PairedDistractingDMC
        paired = env_id.startswith('PairedDistractingDMC-')
        # 格式: DistractingDMC-[difficulty]-[dynamic|static]-domain_task
        # 例如: DistractingDMC-easy-static-cartpole_balance
        #       DistractingDMC-easy-dynamic-cartpole_balance
        #       DistractingDMC-easy-cartpole_balance   (默认static)
        #       DistractingDMC-cartpole_balance         (默认easy+static)
        #       DistractingDMC-custom-static-cartpole_balance  (读 config/distract.yaml 自定义)
        parts = env_id.split('-', maxsplit=3)
        difficulty = 'easy'
        dynamic = False
        name = parts[-1].lower()
        for token in parts[1:-1]:
            if token in ('easy', 'medium', 'hard', 'custom'):
                difficulty = token
            elif token == 'dynamic':
                dynamic = True
            elif token == 'static':
                dynamic = False
        custom_kwargs = None
        if difficulty == 'custom':
            # Fine-grained distraction params from config (config/distract.yaml).
            custom_kwargs = dict(
                scale=getattr(conf, 'distract_scale', 0.1),
                num_videos=getattr(conf, 'distract_num_videos', 4),
                video_alpha=getattr(conf, 'distract_video_alpha', 1.0),
                ground_alpha=getattr(conf, 'distract_ground_alpha', 1.0),
                background=getattr(conf, 'distract_background', True),
                camera=getattr(conf, 'distract_camera', True),
                color=getattr(conf, 'distract_color', True),
            )
        cls = PairedDistractingDMC if paired else DistractingDMC
        env = cls(name, action_repeat=env_action_repeat,
                  difficulty=difficulty, dynamic=dynamic,
                  custom_kwargs=custom_kwargs,
                  gt_mask=getattr(conf, 'use_gt_mask', False),
                  gt_mask_supersample=getattr(conf, 'gt_mask_supersample', 1))

    elif env_id.startswith('DMC-'):
        from .dmc import DMC
        env = DMC(env_id.split('-', maxsplit=1)[1].lower(), action_repeat=env_action_repeat,
                  gt_mask=getattr(conf, 'use_gt_mask', False),
                  gt_mask_supersample=getattr(conf, 'gt_mask_supersample', 1))
    
    elif env_id.startswith('Embodied-'):
        from .embodied import EmbodiedEnv
        task = env_id.split('-', maxsplit=1)[1].lower()
        env = EmbodiedEnv(task, action_repeat=env_action_repeat, time_limit=env_time_limit)
        env_time_limit = 0  # This is handled by embodied.Env

    else:
        env = gym.make(env_id)
        env = DictWrapper(env)

    if hasattr(env.action_space, 'n'):
        env = OneHotActionWrapper(env)
    if env_time_limit > 0:
        env = TimeLimitWrapper(env, env_time_limit)
    env = ActionRewardResetWrapper(env, no_terminal)
    env = CollectWrapper(env)
    return env
