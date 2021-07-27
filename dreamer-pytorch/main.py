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


def build_and_train(log_dir, task="WipeDesk", environments=RLBench, run_ID=0, cuda_idx=0, eval=False,
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
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=100 / action_repeat)]
    )
    # with these wrapper_kwargs
    # you'll have: TimeLimit(NormalizeActions(ActionRepeat(RLBench,
    #                        dict(amount=action_repeat),
    #                                         dict(),
    #                                                       dict(amount=action_repeat))
    environments_args = {}
    environments_eval_args = {}
    if environments == RLBench:
        environments_args = {"config": {}}  # {task: task}}  # , "_env": ""}}
        environments_eval_args = {"config": {}}  # "task": task}
    else:
        print(environments)
    print(environments, RLBench, environments_args)

    eval_n_envs = 0
    if eval:
        eval_n_envs = 1

    sampler = SerialSampler(
        # TODO: kwargs are difficult to debug, prefer to put the parameters here
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        eval_CollectorCls=SerialEvalCollector,
        env_kwargs=environments_args,
        eval_env_kwargs=environments_eval_args,
        # samples are temporarily stored in memory, when there are batch_T samples,
        # they are transferred to the replay buffer
        batch_T=1,
        # number of environment instances to run (in parallel), becomes second batch dimension
        batch_B=1,
        # if taking random number of steps before start of training, to decorrelate batch states:
        max_decorrelation_steps=0,
        # number of environment instances for agent evaluation (0 for no separate evaluation)
        # (requires a parallel sampler)
        # must be set to 1 if running with --eval
        eval_n_envs=eval_n_envs,
        # max total number of steps (time * n_envs) per evaluation call
        eval_max_steps=int(10e3),
        # Optional earlier cutoff for evaluation phase (note that this shouldn't be the imagination
        # phase, which needs long horizons(?))
        eval_max_trajectories=5,
    )

    # the default setting are as they were used by Hafner, except kl_scale
    # which is set to 0.1 in J.Frost's implementation by default and to 10
    # in Hafner's implementation
    algo = Dreamer(
        batch_size=50,
        batch_length=50,
        train_every=1000,  # Imagination every
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
        # replay_ratio=8,  # never used
        # n_step_return=1,  # never used

        updates_per_sync=1,  # ? For async mode only. (not implemented)
        free_nats=3,  # PlaNet (1811.04551). pp:12" We do not scale the KL divergence terms relative to the
        # reconstruction terms but grant the model 3 free nats by clipping the divergence loss below this value.
        # In a previous version of the agent, we used latent overshooting and an additional fixed global prior, but we
        # found  this not to be necessary."
        kl_scale=1.0,  # in J.Frost: 0.1 (may be due to PyTorch, maybe it is multiplied by 10 somewhere) -- see
        # free_nats "... we do not scale the KL divergence.."
        type=torch.float,
        prefill=5000,  # when a fresh run starts, the replay buffer is filled with prefill samples (itrs.)
        log_video=True,  # requires moviepy (seems to not always work, may be due to TF dependencies)
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
        expl_decay=None
        # should we add the rest here? (if so, need to step through the trace)
        # ModelCls=SWDDreamerModel,  # this has many params
        # initial_model_state_dict=agent_state_dict,
        # ModelCls=AgentModel,
        # model_kwargs=None,
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
    config = {"task": task}
    name = "dreamer_" + task
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', help='RLBench task', default='WipeDesk')
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
