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
# from dreamer.envs.dmc import DeepMindControl
# from dreamer.envs.atari import Atari
# from dreamer.envs.rlbench import RLBench
from dreamer.envs.imitation import RLBench
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from rlpyt.samplers.serial.collectors import SerialEvalCollector

from dreamer.envs.time_limit import TimeLimit


def build_and_train(log_dir, task="FastSingle2xtarget", environments=RLBench, run_ID=0, cuda_idx=0, eval=False,  #
                    save_model='last', load_model_path=None):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = 2

    factory_method = make_wapper(
        base_class=environments,
        # wrapper_classes: list of wrapper classes in order inner-first, outer-last
        wrapper_classes=[ActionRepeat, NormalizeActions, TimeLimit],
        # list of kwargs dictionaries passed to the wrapper classes:
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=100 / action_repeat)])
    # you'll have: TimeLimit(NormalizeActions(ActionRepeat(RLBench,
    #                        dict(amount=action_repeat),
    #                                         dict(),
    #                                                       dict(amount=action_repeat))
    # so, how to pass arguments to base_class?
    environments_args = {}
    environments_eval_args = {}
    #    if environments == DeepMindControl:
    #        environments_args = {"name": task}
    #       environments_eval_args = {"name": task}
    if environments == RLBench:
        environments_args = {"config": {}}  # {task: task}}  # , "_env": ""}}
        environments_eval_args = {"config": {}}  # "task": task}
    #    if isinstance(environments, Atari):
    #        environments_args = dict(name=task)
    #        environments_eval_args = dict(name=task)
    else:
        print(environments)
    print(environments, RLBench, environments_args)

    eval_n_envs=0
    if eval:
        eval_n_envs=1

    batching_number = 1
    print("batching number is set to {}: see main.py".format(batching_number))

    sampler = SerialSampler(
        # kwargs are difficult to debug, prefer to put the parameters here
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        # when running with --eval, this seems to be missing somewhere but
        # this is not a fix (it doesn't work)
        eval_CollectorCls=SerialEvalCollector,
        # env_kwargs allows passing the arguments to (JUST?) base_class: so base_class is a poor name choice,
        # base_class should be named environment_class,
        # unfortunately, SerialSampler is defined in RLPyt, we can replicate and overwrite it in this repo
        # - to get rid of **kwargs
        # - pass the env_kwargs to base_class in the factory method too
        # - don't split SerialSampler in a super(BaseSampler) and inherited (SerialSampler) class
        #     unless useful
        env_kwargs=environments_args,
        eval_env_kwargs=environments_eval_args,
        batch_T=batching_number,
        # number of environment instances to run (in parallel), becomes second batch dimension
        batch_B=1,
        # if taking random number of steps before start of training, to decorrelate batch states:
        max_decorrelation_steps=0,
        # number of environment instances for agent evaluation (0 for no separate evaluation)
        # (sounds like it requires a parallel sampler)
        # must be set  to 1 if running with --eval
        # --eval may work when running with VirtualGL, throws a Qt Error when running on Desktop
        eval_n_envs=eval_n_envs,
        # max total number of steps (time * n_envs) per evaluation call
        eval_max_steps=int(10e3),
        # Optional earlier cutoff for evaluation phase (note that this shouldn't be the imagination phase, which needs long horizons(?))
        eval_max_trajectories=5,
    )

    algo = Dreamer(
        batch_size=50,
        batch_length=50,
        train_every=1000/batching_number,
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
        replay_ratio=8,  # !D
        n_step_return=1,  #!D

        updates_per_sync=1,  #? For async mode only. (not implemented)
        free_nats=3,
        kl_scale=1.0,  # here: 0.1
        type=torch.float,
        prefill=5000/batching_number,
        log_video=True,
        video_every=int(1e1),
        video_summary_t=25,
        video_summary_b=4,
        use_pcont=False,
        pcont_scale=10.0,
    )
    agent = BenchmarkDreamerAgent2(
        train_noise=0.3,
        eval_noise=0,
        expl_type="additive_gaussian",
        expl_min=None,
        expl_decay=None#,
        # should we add the rest here? (if so, need to step through the trace)
        #ModelCls=AtariDreamerModel,  # this has many params
        #initial_model_state_dict=agent_state_dict,
        #ModelCls=AgentModel,
        #model_kwargs=None,
    )
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,  #  = n_steps / batching_number
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    relevant_parameter_settings = ""  # now using dense reward, single front 64 camera and large target object/no distractor target reach by default
    config = {"task": task}
    name = "dreamer_" + task + "_" + relevant_parameter_settings
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', help='task or (Atari) game', default='FastSingle2xtarget')
    parser.add_argument('--environments', help='Environments (class) to use', default='RLBench')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-model', help='save model', type=str, default='last',
                        choices=['all', 'none', 'gap', 'last'])
    parser.add_argument('--load-model-path', help='load (.pkl) model from path', type=str)  # path to params.pkl

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
