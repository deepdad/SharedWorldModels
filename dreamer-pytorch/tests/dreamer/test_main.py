import datetime
import os
import argparse
import torch
from rlpyt.samplers.collections import TrajInfo

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.benchmark_dreamer_agent import BenchmarkDreamerAgent2
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.wrapper import make_wapper
# from dreamer.envs.rlbench import RLBench
from dreamer.envs.imitation import RLBench
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from rlpyt.samplers.serial.collectors import SerialEvalCollector

from dreamer.envs.time_limit import TimeLimit


def build_and_train(log_dir="./", task="FastSingle2xtarget", environments=RLBench, run_ID=0, cuda_idx=0, eval=False,
                    save_model='last', load_model_path=None):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = 2

    factory_method = make_wapper(
        base_class=environments,
        wrapper_classes=[ActionRepeat, NormalizeActions, TimeLimit],
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)]
    )
    environments_args = {}
    environments_eval_args = {}
    if environments == RLBench:
        environments_args = {"config": {}}
        environments_eval_args = {"config": {}}
    else:
        print(environments)
    print(environments, RLBench, environments_args)

    eval_n_envs = 0
    if eval:
        eval_n_envs = 1

    sampler = SerialSampler(
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        eval_CollectorCls=SerialEvalCollector,
        env_kwargs=environments_args,
        eval_env_kwargs=environments_eval_args,
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=eval_n_envs,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    algo = Dreamer(
        batch_size=1,
        batch_length=5,
        train_every=10,
        train_steps=2,
        pretrain=100,
        model_lr=6e-4,
        value_lr=8e-5,
        actor_lr=8e-5,
        grad_clip=100.0,
        dataset_balance=False,
        discount=0.99,
        discount_lambda=0.95,
        horizon=5,
        action_dist='tanh_normal',
        action_init_std=5.0,
        expl='additive_gaussian',
        expl_amount=0.3,
        expl_decay=0.0,
        expl_min=0.0,
        OptimCls=torch.optim.Adam,
        optim_kwargs=None,
        initial_optim_state_dict=optimizer_state_dict,

        replay_size=100,
        updates_per_sync=1,
        free_nats=3,
        kl_scale=0.1,
        type=torch.float,
        prefill=10,
        log_video=False,
        video_every=int(1e1),
        video_summary_t=25,
        video_summary_b=4,
        use_pcont=True,
        pcont_scale=10.0,
    )
    agent = BenchmarkDreamerAgent2(
        train_noise=0.3,
        eval_noise=0,
        expl_type="additive_gaussian",
        expl_min=0.1,
        expl_decay=2000 / 0.3,
    )
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=20,
        log_interval_steps=10,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = {"task": task}
    name = "dreamer_" + task
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


def test_main():
    build_and_train()
