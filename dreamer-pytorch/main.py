import datetime
import os
import argparse
import torch
from rlpyt.samplers.collections import TrajInfo

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context

from rlbench.action_modes import ArmActionMode, ActionMode
from sampler.shared_wm_sampler import SharedWorldModelSampler
from dreamer.agents.benchmark_dreamer_agent import BenchmarkDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.wrapper import make_wapper
from dreamer.envs.rlbench import RLBench
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.time_limit import TimeLimit


def build_and_train(log_dir, task="TargetReach", environments=RLBench, run_ID=0, cuda_idx=0, eval=False,  #
                    save_model='last', load_model_path=None):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = 2

    factory_method = make_wapper(
        base_class=environments,
        wrapper_classes=[ActionRepeat, NormalizeActions, TimeLimit],
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=100 / action_repeat)])
    environments_args = {}
    if environments == RLBench:
        environments_args = {"config": {"action_mode": ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)}}
        s_environments_args = {"config": {"action_mode": ActionMode(ArmActionMode.ABS_JOINT_POSITION)}}

    else:
        print(environments)

    print(environments, RLBench, environments_args)


    sampler = SharedWorldModelSampler(
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        first_env_kwargs=environments_args,
        second_env_kwargs=s_environments_args,
        env_change_itr=500,
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0
    )

    algo = Dreamer(
        batch_size=35,
        batch_length=35,
        train_every=1000,
        train_steps=100,
        pretrain=100,
        model_lr=6e-4,
        value_lr=8e-5,
        actor_lr=8e-5,
        grad_clip=100.0,
        dataset_balance=False,
        discount=0.99,
        discount_lambda=0.95,
        horizon=15,
        action_dist='tanh_normal',
        action_init_std=5.0,
        expl='additive_gaussian',
        expl_amount=0.3,
        expl_decay=0.0,
        expl_min=0.0,
        OptimCls=torch.optim.Adam,
        optim_kwargs=None,
        initial_optim_state_dict=optimizer_state_dict,
        replay_size=int(5e5),
        replay_ratio=8,
        n_step_return=1,
        updates_per_sync=1,  # For async mode only. (not implemented)
        free_nats=3,
        kl_scale=1,
        type=torch.float,
        prefill=5000,
        log_video=True,
        video_every=int(1e1),
        video_summary_t=25,
        video_summary_b=4,
        use_pcont=False,
        pcont_scale=10.0,
    )
    agent = BenchmarkDreamerAgent(
        train_noise=0.3,
        eval_noise=0,
        expl_type="additive_gaussian",
        expl_min=None,
        expl_decay=None
    )
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    relevant_parameter_settings = "singlefront_64camera_densereward"
    config = {"task": task}
    name = "dreamer_" + task + "_" + relevant_parameter_settings
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', help='task or (Atari) game', default='TargetReach')
    parser.add_argument('--environments', help='Environments (class) to use', default='RLBench')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-model', help='save model', type=str, default='last',
                        choices=['all', 'none', 'gap', 'last'])
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl

    default_log_dir = os.path.join(
        os.path.dirname(__file__),
        'data',
        'local',
        datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        print(f'run {i} already exists. ')
        i += 1
    print(f'Using run id = {i}')
    if args.environments == "RLBench":
        environments = RLBench
    #    else:
    #        environments = DeepMindControl
    print("Using the {} environments.".format(environments))
    args.run_ID = i
    build_and_train(
        log_dir,
        task=args.task,
        environments=environments,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        eval=args.eval,
        save_model=args.save_model,
        load_model_path=args.load_model_path
    )
