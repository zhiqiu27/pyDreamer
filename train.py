import time
from collections import defaultdict
from datetime import datetime
from logging import critical, debug, error, info, warning
from typing import Iterator, Optional

import mlflow
import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader

from multiprocessing import Process

import generator as generator_module
from pydreamer import tools
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from pydreamer.models import *
from pydreamer.models.functions import map_structure, nanmean
from pydreamer.models.masker import ObservationMasker
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from pydreamer.tools import *


def run(conf):

    configure_logging(prefix='[TRAIN]')
    mlrun = mlflow_init()
    wandb_init()
    artifact_uri = mlrun.info.artifact_uri

    torch.distributions.Distribution.set_default_validate_args(False)
    torch.backends.cudnn.benchmark = True  # type: ignore
    device = torch.device(conf.device)
    
    # Data directories

    if conf.offline_data_dir:
        online_data = False
        input_dirs = to_list(conf.offline_data_dir)
    else:
        online_data = True
        input_dirs = [
            f'{artifact_uri}/episodes/{i}'
            for i in range(max(conf.generator_workers_train, conf.generator_workers))
        ]
    
    if conf.offline_prefill_dir:
        input_dirs.extend(to_list(conf.offline_prefill_dir))

    if conf.offline_eval_dir:
        eval_dirs = to_list(conf.offline_eval_dir)
    else:
        eval_dirs = [
            f'{artifact_uri}/episodes_eval/{i}'
            for i in range(max(conf.generator_workers_eval, conf.generator_workers))
        ]

    if conf.offline_test_dir:
        test_dirs = to_list(conf.offline_test_dir)
    else:
        test_dirs = eval_dirs

    # Wait for prefill

    if online_data:
        while True:
            data_train_stats = DataSequential(MlflowEpisodeRepository(input_dirs), conf.batch_length, conf.batch_size, check_nonempty=False)
            mlflow_log_metrics({
                'train/data_steps': data_train_stats.stats_steps,
                'train/data_env_steps': data_train_stats.stats_steps * conf.env_action_repeat,
                '_timestamp': datetime.now().timestamp(),
            }, step=0)
            if data_train_stats.stats_steps < conf.generator_prefill_steps:
                debug(f'Waiting for prefill: {data_train_stats.stats_steps}/{conf.generator_prefill_steps} steps...')
                time.sleep(10)
            else:
                info(f'Done prefilling: {data_train_stats.stats_steps}/{conf.generator_prefill_steps} steps.')
                break

        if data_train_stats.stats_steps * conf.env_action_repeat >= conf.n_env_steps:
            # Prefill-only job, or resumed already finished job
            info(f'Finished {conf.n_env_steps} env steps.')
            return

    # Data reader

    data = DataSequential(MlflowEpisodeRepository(input_dirs),
                          conf.batch_length,
                          conf.batch_size,
                          skip_first=True,
                          reload_interval=120 if online_data else 0,
                          buffer_size=conf.buffer_size if online_data else conf.buffer_size_offline,
                          reset_interval=conf.reset_interval,
                          allow_mid_reset=conf.allow_mid_reset)
    preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                              image_key=conf.image_key,
                              map_categorical=conf.map_channels if conf.map_categorical else None,
                              map_key=conf.map_key,
                              action_dim=conf.action_dim,
                              clip_rewards=conf.clip_rewards,
                              amp=conf.amp and device.type == 'cuda')

    # MODEL

    if conf.model == 'dreamer':
        model = Dreamer(conf)
    else:
        model: Dreamer = WorldModelProbe(conf)  # type: ignore
    model.to(device)
    print(model)
    # print(repr(model))
    mlflow_log_text(repr(model), 'architecture.txt')

    optimizers = model.init_optimizers(conf.adam_lr, conf.adam_lr_actor, conf.adam_lr_critic, conf.adam_eps)
    resume_step = tools.mlflow_load_checkpoint(model, optimizers)
    if resume_step:
        info(f'Loaded model from checkpoint epoch {resume_step}')

    # Load pre-trained masker (lives in a separate MLflow run from the main training run)
    masker_run_id = getattr(conf, 'masker_run_id', '')
    masker_full_sd = None  # full pre-training state_dict (mask_net + aux heads), reused for refresh
    if getattr(conf, 'use_masker', False) and model.wm.masker is not None:
        if masker_run_id:
            import mlflow
            from mlflow.tracking.client import MlflowClient
            import tempfile as _tmpfile
            client = MlflowClient()
            with _tmpfile.TemporaryDirectory() as tmpdir:
                path = client.download_artifacts(masker_run_id, 'checkpoints/masker.pt', tmpdir)
                ckpt = torch.load(path, map_location=device)
                masker_full_sd = ckpt['masker_state_dict']
                # strict=False: the pre-training checkpoint also holds aux-head weights
                # (aux_encoder/inverse_head) that the inference-only masker in the
                # WorldModel does not have. mask_net keys must all be present though.
                missing, _ = model.wm.masker.load_state_dict(masker_full_sd, strict=False)
                assert not missing, f'Masker checkpoint missing keys: {missing}'
                info(f'Loaded masker from mlflow run {masker_run_id} (epoch {ckpt["epoch"]})')
        else:
            info('WARNING: use_masker=True but masker_run_id not set, masker weights are random')
        # Note: the masker is already frozen (if masker_frozen) in WorldModel.__init__,
        # before the optimizers above were built, so it is excluded from the param group.
        if getattr(conf, 'masker_frozen', True):
            info('Masker frozen (excluded from optimizer updates)')

    # Persistent trainable masker for periodic on-policy refresh.
    # It lives outside `model` (own aux heads + own optimizer) and its mask_net is
    # copied into the frozen model.wm.masker at each sync, so the world-model masker
    # co-evolves with the policy without ever entering the main optimizer.
    masker_trainer = None
    masker_opt = None
    masker_pretrain_dirs = []
    masker_collect = None  # in-flight paired-collection state, or None when idle
    if getattr(conf, 'use_masker', False) and getattr(conf, 'masker_refresh', False) and model.wm.masker is not None:
        if not getattr(conf, 'masker_frozen', True):
            warning('masker_refresh=True but masker_frozen=False: the world-model masker is in the '
                    'main optimizer, so periodic sync will fight gradient updates. Set masker_frozen=True.')
        masker_trainer = ObservationMasker(
            in_channels=conf.image_channels,
            action_dim=conf.action_dim,
            depth=getattr(conf, 'masker_depth', 32),
            actor_dist=conf.actor_dist,
            inference_only=False,  # keep aux heads so training_step() works
        ).to(device)
        if masker_full_sd is not None:
            masker_trainer.load_state_dict(masker_full_sd)  # full: mask_net + aux heads (warm-started)
        masker_opt = torch.optim.AdamW(masker_trainer.parameters(),
                                       lr=getattr(conf, 'masker_refresh_lr', 1e-4),
                                       eps=conf.adam_eps)
        # Pre-training paired data, mixed in during fine-tune to avoid forgetting broad coverage.
        if getattr(conf, 'masker_refresh_mix_pretrain', True) and masker_run_id:
            try:
                from mlflow.tracking.client import MlflowClient
                pre_uri = MlflowClient().get_run(masker_run_id).info.artifact_uri
                masker_pretrain_dirs = [f'{pre_uri}/episodes/{i}' for i in range(conf.generator_workers)]
            except Exception as e:
                warning(f'Masker refresh: could not resolve pre-training paired dirs: {repr(e)}')
        info(f'Masker periodic refresh enabled '
             f'(every {conf.masker_refresh_every_ckpt} ckpts, collect {conf.masker_refresh_collect_steps} steps, '
             f'finetune {conf.masker_refresh_train_steps} steps @ lr {getattr(conf, "masker_refresh_lr", 1e-4)})')

    # ---------------------
    # TRAINING
    # ---------------------

    start_time = time.time()
    steps = resume_step or 0
    last_time = start_time
    last_steps = steps
    metrics = defaultdict(list)
    metrics_max = defaultdict(list)

    timers = {}

    def timer(name, verbose=False):
        if name not in timers:
            timers[name] = Timer(name, verbose)
        return timers[name]

    states = {}  # by worker
    data_iter = iter(DataLoader(WorkerInfoPreprocess(preprocess(data)),
                                batch_size=None,
                                num_workers=conf.data_workers,
                                prefetch_factor=20 if conf.data_workers else None,  # GCS download has to be shorter than this many batches (e.g. 1sec < 20*300ms)
                                pin_memory=True))

    scaler = GradScaler(enabled=conf.amp)

    with get_profiler(conf) as profiler:
        while True:
            with timer('total'):
                profiler.step()
                steps += 1
                will_log_batch = steps % conf.logbatch_interval == 1
                will_image_pred = (
                    will_log_batch or
                    steps % conf.log_interval >= int(conf.log_interval * 0.9)  # 10% of batches
                )

                # Make batch

                with timer('data'):

                    batch, wid = next(data_iter)
                    obs: Dict[str, Tensor] = map_structure(batch, lambda x: x.to(device))  # type: ignore

                # Forward

                with timer('forward'):
                    with autocast(enabled=conf.amp):

                        state = states.get(wid)
                        if state is None:
                            state = model.init_state(conf.batch_size * conf.iwae_samples)
                        losses, new_state, loss_metrics, tensors, dream_tensors = \
                            model.training_step(
                                obs,
                                state,
                                do_image_pred=will_image_pred,
                                do_dream_tensors=will_log_batch)
                        if conf.keep_state:
                            states[wid] = new_state

                # Backward

                with timer('backward'):

                    for opt in optimizers:
                        opt.zero_grad()
                    for i, loss in enumerate(losses):
                        retain = (i < len(losses) - 1)
                        scaler.scale(loss).backward(retain_graph=retain)  # type: ignore

                # Grad step

                with timer('gradstep'):  # CUDA wait happens here

                    for opt in optimizers:
                        scaler.unscale_(opt)
                    grad_metrics = model.grad_clip(conf.grad_clip, conf.grad_clip_ac)
                    for opt in optimizers:
                        scaler.step(opt)
                    scaler.update()

                with timer('other'):

                    # Metrics

                    for k, v in loss_metrics.items():
                        if not np.isnan(v.item()):
                            metrics[k].append(v.item())
                    for k, v in grad_metrics.items():
                        if np.isfinite(v.item()):  # It's ok for grad norm to be inf, when using amp
                            metrics[k].append(v.item())
                            metrics_max[k].append(v.item())
                    for k in ['reward', 'reset', 'terminal']:
                        metrics[f'data_{k}'].append(batch[k].float().mean().item())
                    for k in ['reward']:
                        metrics_max[f'data_{k}'].append(batch[k].max().item())

                    # Log sample

                    if will_log_batch:
                        log_batch_npz(batch, tensors, f'{steps:07}.npz', subdir='d2_wm_closed')
                    if dream_tensors:
                        log_batch_npz(batch, dream_tensors, f'{steps:07}.npz', subdir='d2_wm_dream')

                    # Log data buffer size

                    if online_data and steps % conf.logbatch_interval == 0:
                        data_train_stats = DataSequential(MlflowEpisodeRepository(input_dirs), conf.batch_length, conf.batch_size)
                        metrics['data_steps'].append(data_train_stats.stats_steps)
                        metrics['data_env_steps'].append(data_train_stats.stats_steps * conf.env_action_repeat)
                        if data_train_stats.stats_steps * conf.env_action_repeat >= conf.n_env_steps:
                            info(f'Finished {conf.n_env_steps} env steps.')
                            return

                    # Log metrics

                    if steps % conf.log_interval == 0:
                        metrics = {f'train/{k}': np.array(v).mean() for k, v in metrics.items()}
                        metrics.update({f'train/{k}_max': np.array(v).max() for k, v in metrics_max.items()})
                        metrics['train/steps'] = steps
                        metrics['_step'] = steps
                        metrics['_loss'] = metrics.get('train/loss_model', 0)
                        metrics['_timestamp'] = datetime.now().timestamp()

                        t = time.time()
                        fps = (steps - last_steps) / (t - last_time)
                        metrics['train/fps'] = fps
                        last_time, last_steps = t, steps

                        info(f"[{steps:06}]"
                             f"  loss_model: {metrics.get('train/loss_model', 0):.3f}"
                             f"  loss_critic: {metrics.get('train/loss_critic', 0):.3f}"
                             f"  policy_value: {metrics.get('train/policy_value',0):.3f}"
                             f"  policy_entropy: {metrics.get('train/policy_entropy',0):.3f}"
                             f"  fps: {metrics['train/fps']:.3f}"
                             )
                        if steps > conf.log_interval:  # Skip the first batch, because the losses are very high and mess up y axis
                            mlflow_log_metrics(metrics, step=steps)
                        metrics = defaultdict(list)
                        metrics_max = defaultdict(list)

                    # Save model

                    if steps % conf.save_interval == 0:
                        tools.mlflow_save_checkpoint(model, optimizers, steps)
                        info(f'Saved model checkpoint {steps}')

                    # Masker periodic refresh: collect on-policy paired data in the
                    # background, then fine-tune the masker and sync it in-place.

                    if masker_trainer is not None:
                        masker_collect = masker_refresh_step(
                            conf, steps, artifact_uri, device, model,
                            masker_trainer, masker_opt, masker_pretrain_dirs, masker_collect)

                    # Stop

                    if steps >= conf.n_steps:
                        info(f'Finished {conf.n_steps} grad steps.')
                        return

                # Evaluate

                with timer('eval'):
                    if conf.eval_interval and steps % conf.eval_interval == 0:
                        try:
                            # Test = same settings as train
                            data_test = DataSequential(MlflowEpisodeRepository(test_dirs), conf.batch_length, conf.test_batch_size, skip_first=False, reset_interval=conf.reset_interval)
                            test_iter = iter(DataLoader(preprocess(data_test), batch_size=None))
                            evaluate('test', steps, model, test_iter, device, conf.test_batches, conf.iwae_samples, conf.keep_state, conf.test_save_size, conf)

                            # Eval = no state reset, multisampling
                            data_eval = DataSequential(MlflowEpisodeRepository(eval_dirs), conf.batch_length, conf.eval_batch_size, skip_first=False)
                            eval_iter = iter(DataLoader(preprocess(data_eval), batch_size=None))
                            evaluate('eval', steps, model, eval_iter, device, conf.eval_batches, conf.eval_samples, True, conf.eval_save_size, conf)

                        except Exception as e:
                            # This catch is useful if there is no eval data generated yet
                            warning(f'Evaluation failed: {repr(e)}')

            for k, v in timers.items():
                metrics[f'timer_{k}'].append(v.dt_ms)

            if conf.verbose:
                info(f"[{steps:06}] timers"
                     f"  TOTAL: {timer('total').dt_ms:>4}"
                     f"  data: {timer('data').dt_ms:>4}"
                     f"  forward: {timer('forward').dt_ms:>4}"
                     f"  backward: {timer('backward').dt_ms:>4}"
                     f"  gradstep: {timer('gradstep').dt_ms:>4}"
                     f"  eval: {timer('eval').dt_ms:>4}"
                     f"  other: {timer('other').dt_ms:>4}"
                     )


def evaluate(prefix: str,
             steps: int,
             model: Dreamer,
             data_iterator: Iterator,
             device,
             eval_batches: int,
             eval_samples: int,
             keep_state: bool,
             save_size: int,
             conf):

    start_time = time.time()
    metrics_eval = defaultdict(list)
    state = None
    tensors = None
    npz_datas = []
    n_finished_episodes = np.zeros(1)
    do_output_tensors = True

    for i_batch in range(eval_batches):
        with torch.no_grad():

            batch = next(data_iterator)
            obs: Dict[str, Tensor] = map_structure(batch, lambda x: x.to(device))  # type: ignore
            T, B = obs['action'].shape[:2]

            if i_batch == 0:
                info(f'Evaluation ({prefix}): batches: {eval_batches},  size(T,B,I): ({T},{B},{eval_samples})')

            reset_episodes = obs['reset'].any(dim=0)  # (B,)
            n_reset_episodes = reset_episodes.sum().item()
            n_continued_episodes = (~reset_episodes).sum().item()
            if i_batch == 0:
                n_finished_episodes = np.zeros(B)
            else:
                n_finished_episodes += reset_episodes.cpu().numpy()

            # Log _last predictions from the last batch of previous episode # TODO: make generic for goal probes

            if n_reset_episodes > 0 and tensors is not None and 'loss_map' in tensors:
                logprob_map_last = (tensors['loss_map'].mean(dim=0) * reset_episodes).sum() / reset_episodes.sum()
                metrics_eval['logprob_map_last'].append(logprob_map_last.item())

            # Open loop & unseen logprob

            if n_continued_episodes > 0:
                with autocast(enabled=conf.amp):
                    _, _, _, tensors_im, _ = \
                        model.training_step(obs,  # observation will be ignored in forward pass because of imagine=True
                                            state,
                                            iwae_samples=eval_samples,
                                            imag_horizon=conf.imag_horizon,
                                            do_open_loop=True,
                                            do_image_pred=True)

                    if np.random.rand() < 0.10:  # Save a small sample of batches
                        r = obs['reward'].sum().item()
                        log_batch_npz(batch, tensors_im, f'{steps:07}_{i_batch}_r{r:.0f}.npz', subdir=f'd2_wm_open_{prefix}')

                    mask = (~reset_episodes).float()
                    for key, logprobs in tensors_im.items():
                        if key.startswith('logprob_'):  # logprob_image, logprob_reward, ...
                            # Many logprobs will be nans - that's fine. Just take mean of those tahat exist
                            lps = logprobs[:5] * mask / mask  # set to nan where ~mask
                            lp = nanmean(lps).item()
                            if not np.isnan(lp):
                                metrics_eval[f'{key}_open'].append(lp)  # logprob_image_open, ...

            # Closed loop & loss

            with autocast(enabled=conf.amp):
                if state is None or not keep_state:
                    state = model.init_state(B * eval_samples)

                _, state, loss_metrics, tensors, _ = \
                    model.training_step(obs,
                                        state,
                                        iwae_samples=eval_samples,
                                        imag_horizon=conf.imag_horizon,
                                        do_image_pred=True)

                for k, v in loss_metrics.items():
                    if not np.isnan(v.item()):
                        metrics_eval[k].append(v.item())

            # Log one episode batch

            if do_output_tensors:
                npz_datas.append(prepare_batch_npz(dict(**batch, **tensors), take_b=save_size))
            if n_finished_episodes[0] > 0:
                # log predictions until first episode is finished
                do_output_tensors = False

    metrics_eval = {f'{prefix}/{k}': np.array(v).mean() for k, v in metrics_eval.items()}
    mlflow_log_metrics(metrics_eval, step=steps)

    if len(npz_datas) > 0:
        npz_data = {k: np.concatenate([d[k] for d in npz_datas], 1) for k in npz_datas[0]}
        print_once(f'Saving batch d2_wm_closed_{prefix}: ', {k: tuple(v.shape) for k, v in npz_data.items()})
        r = npz_data['reward'][0].sum().item()
        tools.mlflow_log_npz(npz_data, f'{steps:07}_r{r:.0f}.npz', subdir=f'd2_wm_closed_{prefix}', verbose=True)

    info(f'Evaluation ({prefix}): done in {(time.time()-start_time):.0f} sec, recorded {n_finished_episodes.sum()} episodes')


def log_batch_npz(batch: Dict[str, Tensor],
                  tensors: Dict[str, Tensor],
                  filename: str,
                  subdir: str):

    data = dict(**batch, **tensors)
    print_once(f'Saving batch {subdir} (input): ', {k: tuple(v.shape) for k, v in data.items()})
    data = prepare_batch_npz(data)
    print_once(f'Saving batch {subdir} (proc.): ', {k: tuple(v.shape) for k, v in data.items()})
    tools.mlflow_log_npz(data, filename, subdir, verbose=True)


def prepare_batch_npz(data: Dict[str, Tensor], take_b=999):

    def unpreprocess(key: str, val: Tensor) -> np.ndarray:
        if take_b < val.shape[1]:
            val = val[:, :take_b]

        x = val.cpu().numpy()  # (T,B,*)
        if x.dtype in [np.float16, np.float64]:
            x = x.astype(np.float32)

        if len(x.shape) == 2:  # Scalar
            pass

        elif len(x.shape) == 3:  # 1D vector
            pass

        elif len(x.shape) == 4:  # 2D tensor
            pass

        elif len(x.shape) == 5:  # 3D tensor - image
            if not (x.dtype == np.float32 and (key.startswith('image') or key.startswith('map'))):
                return None  # skip unexpected 3D tensors (e.g. 'pixels' from distracting_control)


            if x.shape[-1] == x.shape[-2]:  # (T,B,C,W,W)
                x = x.transpose(0, 1, 3, 4, 2)  # => (T,B,W,W,C)
            assert x.shape[-2] == x.shape[-3], 'Assuming rectangular images, otherwise need to improve logic'

            if x.shape[-1] in [1, 3]:
                # RGB or grayscale
                x = ((x + 0.5) * 255.0).clip(0, 255).astype('uint8')
            elif np.allclose(x.sum(axis=-1), 1.0) and np.allclose(x.max(axis=-1), 1.0):
                # One-hot
                x = x.argmax(axis=-1)
            else:
                # Categorical logits
                assert key in ['map_rec', 'image_rec', 'image_pred'], \
                    f'Unexpected 3D categorical logits: {key}: {x.shape}'
                x = scipy.special.softmax(x, axis=-1)

        x = x.swapaxes(0, 1)  # type: ignore  # (T,B,*) => (B,T,*)
        return x

    return {k: v for k, v in ((k, unpreprocess(k, v)) for k, v in data.items()) if v is not None}


def masker_refresh_step(conf, steps, artifact_uri, device, model,
                        masker_trainer, masker_opt, pretrain_dirs, collect):
    """Non-blocking state machine driving one on-policy masker refresh.

    Idle -> (refresh due) spawn paired generators and return collection state.
    Collecting -> poll accumulated steps; once the target is reached (or the
    generators exit), fine-tune the masker, sync it into the world model, and
    return to idle (None). The learner keeps training throughout collection;
    only the short fine-tune pauses it.
    """
    if collect is None:
        due = (steps >= conf.masker_refresh_after_steps and
               steps % (conf.save_interval * conf.masker_refresh_every_ckpt) == 0)
        if due:
            collect = _masker_start_collect(conf, artifact_uri, steps)
            info(f'[masker] started paired collection at step {steps} '
                 f'(target {conf.masker_refresh_collect_steps} steps)')
        return collect

    # Collection in flight - poll only occasionally (listing the repo is not free).
    if steps % conf.log_interval != 0:
        return collect

    collected = DataSequential(MlflowEpisodeRepository(collect['dirs']),
                               conf.batch_length, conf.batch_size,
                               check_nonempty=False).stats_steps
    alive = any(p.is_alive() for p in collect['procs'])
    if collected < conf.masker_refresh_collect_steps and alive:
        debug(f'[masker] collecting {collected}/{conf.masker_refresh_collect_steps}')
        return collect

    for p in collect['procs']:
        if p.is_alive():
            p.kill()
    if collected <= 0:
        warning('[masker] no paired data collected; skipping this refresh')
        return None

    _masker_finetune(conf, device, model, masker_trainer, masker_opt,
                     collect['dirs'], pretrain_dirs, steps)
    info(f'[masker] fine-tuned {conf.masker_refresh_train_steps} steps and synced '
         f'at step {steps} ({collected} paired steps collected)')
    return None


def _masker_start_collect(conf, artifact_uri, steps):
    # Unique dirs per burst: the generator compares num_steps against the steps
    # already saved in its save_uri, so reusing a dir across bursts would make
    # every generator after the first exit immediately with nothing collected.
    # Fresh dirs also mean the fine-tune below sees only this burst's on-policy data.
    dirs = [f'{artifact_uri}/episodes_masker/{steps:07}_{i}'
            for i in range(conf.masker_refresh_workers)]
    procs = []
    for i in range(conf.masker_refresh_workers):
        p = Process(
            target=generator_module.main,
            daemon=True,
            kwargs=dict(
                env_id=conf.masker_env_id or conf.env_id,
                save_uri=dirs[i],
                worker_id=900 + i,  # keep clear of the main generator worker ids
                policy_main='network',    # current policy+masker from latest.pt
                policy_prefill='network',
                # Per-worker share of the target: dirs start empty, so the generator
                # exits on its own once its share is saved (kill is only a fallback).
                num_steps=max(1, conf.masker_refresh_collect_steps // conf.masker_refresh_workers),
                num_steps_prefill=0,
                env_time_limit=conf.env_time_limit,
                env_action_repeat=conf.env_action_repeat,
                env_no_terminal=conf.env_no_terminal,
                model_conf=conf,
                log_mlflow_metrics=False,
            ),
        )
        p.start()
        procs.append(p)
    return dict(procs=procs, dirs=dirs)


def _masker_finetune(conf, device, model, masker_trainer, masker_opt,
                     collect_dirs, pretrain_dirs, steps):
    data_dirs = list(collect_dirs)
    if getattr(conf, 'masker_refresh_mix_pretrain', True) and pretrain_dirs:
        data_dirs = data_dirs + list(pretrain_dirs)
    data = DataSequential(MlflowEpisodeRepository(data_dirs),
                          conf.batch_length,
                          conf.batch_size,
                          skip_first=True,
                          buffer_size=conf.buffer_size,
                          reset_interval=conf.reset_interval)
    preprocess = Preprocessor(
        image_categorical=conf.image_channels if conf.image_categorical else None,
        image_key=conf.image_key,
        map_categorical=conf.map_channels if conf.map_categorical else None,
        map_key=conf.map_key,
        action_dim=conf.action_dim,
        clip_rewards=conf.clip_rewards,
        paired_image=True,  # need 'image2' for background-contrast supervision
    )
    data_iter = iter(DataLoader(preprocess(data), batch_size=None, num_workers=0))

    masker_trainer.train()
    log = defaultdict(list)
    for _ in range(conf.masker_refresh_train_steps):
        batch = next(data_iter)
        img_v1 = batch['image'].to(device)
        img_v2 = batch['image2'].to(device)
        action = batch['action'].to(device)
        reset = batch['reset'].to(device)
        loss, metrics = masker_trainer.training_step(
            img_v1, img_v2, action, reset,
            weight_bg=getattr(conf, 'masker_weight_bg', 1.0),
            weight_inv=getattr(conf, 'masker_weight_inv', 1.0),
            mask_temp=getattr(conf, 'masker_mask_temp', 0.5),
            mask_dilate=getattr(conf, 'masker_mask_dilate', 3),
        )
        masker_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(masker_trainer.parameters(), conf.grad_clip)
        masker_opt.step()
        for k, v in metrics.items():
            log[k].append(v.item())

    # Sync the refined mask_net into the frozen world-model masker, in place.
    # requires_grad is untouched (stays frozen), and the masker is not in any
    # optimizer, so no optimizer state has to change.
    ema = getattr(conf, 'masker_refresh_ema', 0.0)
    with torch.no_grad():
        src_sd = masker_trainer.mask_net.state_dict()
        dst = model.wm.masker.mask_net
        if ema and ema > 0.0:
            dst_sd = dst.state_dict()
            for k in dst_sd:
                dst_sd[k].mul_(1.0 - ema).add_(src_sd[k].to(dst_sd[k].device), alpha=ema)
        else:
            dst.load_state_dict(src_sd)

    # Archive the full masker (mask_net + warm aux heads) into the main run.
    tools.mlflow_save_masker_checkpoint(masker_trainer, steps)
    metrics_log = {f'masker_refresh/{k}': np.mean(v) for k, v in log.items()}
    metrics_log['_timestamp'] = datetime.now().timestamp()
    mlflow_log_metrics(metrics_log, step=steps)


def get_profiler(conf):
    if conf.enable_profiler:
        return torch.profiler.profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=3),
            on_trace_ready=tools.tensorboard_trace_handler('./log'),
        )
    else:
        return NoProfiler()
