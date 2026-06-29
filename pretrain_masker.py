import time
from collections import defaultdict
from datetime import datetime
from logging import info
from multiprocessing import Process
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pydreamer import tools
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from pydreamer.models.masker import ObservationMasker
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from pydreamer.tools import *
import generator as generator_module


# ---------------------------------------------------------------------------
# Background augmentation
# ---------------------------------------------------------------------------

def get_background_aug_pair(batch, device) -> Tuple[Tensor, Tensor]:
    """Extract the two background-augmented views produced by PairedDistractingDMC.

    'image'  and 'image2' are already preprocessed to (T, B, C, H, W) float
    by Preprocessor(paired_image=True).

    Returns:
        img_v1: (T, B, C, H, W)  env1 view
        img_v2: (T, B, C, H, W)  env2 view (same foreground, different background)
    """
    return batch['image'].to(device), batch['image2'].to(device)


# ---------------------------------------------------------------------------
# Main pre-training loop
# ---------------------------------------------------------------------------

def run(conf):
    configure_logging(prefix='[PRETRAIN_MASKER]')
    mlrun = mlflow_init()
    wandb_init(conf)
    artifact_uri = mlrun.info.artifact_uri

    device = torch.device(conf.device)

    input_dirs = [
        f'{artifact_uri}/episodes/{i}'
        for i in range(conf.generator_workers)
    ]

    # Launch generator (random policy, collect offline data)

    subprocesses = []
    for i in range(conf.generator_workers):
        info(f'Launching generator {i}')
        p = Process(
            target=generator_module.main,
            daemon=True,
            kwargs=dict(
                env_id=conf.masker_env_id or conf.env_id,
                save_uri=f'{artifact_uri}/episodes/{i}',
                worker_id=i,
                policy_main=conf.generator_prefill_policy,
                policy_prefill=conf.generator_prefill_policy,
                num_steps=conf.masker_data_steps // conf.generator_workers,
                num_steps_prefill=0,
                env_time_limit=conf.env_time_limit,
                env_action_repeat=conf.env_action_repeat,
                env_no_terminal=conf.env_no_terminal,
                model_conf=conf,
                log_mlflow_metrics=False,
            ),
        )
        p.start()
        subprocesses.append(p)

    # Wait for prefill

    while True:
        repo_stats = DataSequential(MlflowEpisodeRepository(input_dirs),
                                    conf.batch_length, conf.batch_size,
                                    check_nonempty=False)
        info(f'Waiting for data: {repo_stats.stats_steps}/{conf.masker_prefill_steps} steps...')
        if repo_stats.stats_steps >= conf.masker_prefill_steps:
            info(f'Data ready: {repo_stats.stats_steps} steps')
            break
        time.sleep(10)

    # Data reader (same setup as train.py)

    data = DataSequential(
        MlflowEpisodeRepository(input_dirs),
        conf.batch_length,
        conf.batch_size,
        skip_first=True,
        reload_interval=120,
        buffer_size=conf.buffer_size,
        reset_interval=conf.reset_interval,
    )
    preprocess = Preprocessor(
        image_categorical=conf.image_channels if conf.image_categorical else None,
        image_key=conf.image_key,
        map_categorical=conf.map_channels if conf.map_categorical else None,
        map_key=conf.map_key,
        action_dim=conf.action_dim,
        clip_rewards=conf.clip_rewards,
        paired_image=True,   # also preprocess 'image2' from PairedDistractingDMC
    )
    data_iter = iter(DataLoader(
        WorkerInfoPreprocess(preprocess(data)),
        batch_size=None,
        num_workers=conf.data_workers,
        prefetch_factor=20 if conf.data_workers else None,
        pin_memory=True,
    ))

    # Masker

    masker = ObservationMasker(
        in_channels=conf.image_channels,
        action_dim=conf.action_dim,
        depth=getattr(conf, 'masker_depth', 32),
        actor_dist=conf.actor_dist,
    ).to(device)
    optimizer = torch.optim.AdamW(masker.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)

    # Training loop

    log_metrics = defaultdict(list)
    steps = 0

    while steps < conf.masker_train_steps:
        steps += 1

        batch, _ = next(data_iter)
        action = batch['action'].to(device)  # (T, B, A)
        reward = batch['reward'].to(device)  # (T, B)
        reset  = batch['reset'].to(device)   # (T, B)

        img_v1, img_v2 = get_background_aug_pair(batch, device)

        # Background-loss warmup: keep weight_bg=0 for the first warmup steps so the
        # inverse/reward heads (with coverage holding the mask high) learn to use the
        # foreground BEFORE the background loss starts pushing the mask down. Without
        # this the area-dominant background collapses the mask to 0 before inverse can
        # exploit it, and once collapsed inverse can never recover.
        bg_warmup = getattr(conf, 'masker_bg_warmup_steps', 1000)
        weight_bg = getattr(conf, 'masker_weight_bg', 1.0) * min(1.0, steps / max(1, bg_warmup))

        loss, metrics = masker.training_step(
            img_v1, img_v2, action, reward, reset,
            weight_bg=weight_bg,
            weight_inv=getattr(conf, 'masker_weight_inv', 1.0),
            weight_reward=getattr(conf, 'masker_weight_reward', 1.0),
            weight_coverage=getattr(conf, 'masker_weight_coverage', 0.1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(masker.parameters(), conf.grad_clip)
        optimizer.step()

        for k, v in metrics.items():
            log_metrics[k].append(v.item())

        if steps % conf.log_interval == 0:
            log = {f'masker/{k}': np.mean(v) for k, v in log_metrics.items()}
            log['_step'] = steps
            log['_timestamp'] = datetime.now().timestamp()
            mlflow_log_metrics(log, step=steps)
            info(f'[{steps:06}]'
                 f'  loss: {np.mean(log_metrics["loss_masker"]):.4f}'
                 f'  bg: {np.mean(log_metrics["loss_bg"]):.4f}'
                 f'  inv: {np.mean(log_metrics["loss_inv"]):.4f}'
                 f'  reward: {np.mean(log_metrics["loss_reward"]):.4f}'
                 f'  mask_mean: {np.mean(log_metrics["mask_mean"]):.3f}')
            log_metrics = defaultdict(list)

        if steps % conf.save_interval == 0:
            tools.mlflow_save_masker_checkpoint(masker, steps)
            info(f'Saved masker checkpoint at step {steps}')

    tools.mlflow_save_masker_checkpoint(masker, steps)
    info(f'Pre-training done.')
    info(f'MLflow run_id: {mlrun.info.run_id}')
    info(f'Pass to main training: --use_masker True --masker_run_id {mlrun.info.run_id}')

    for p in subprocesses:
        p.kill()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from distutils.util import strtobool

    configure_logging('[pretrain_masker]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    run(conf)
