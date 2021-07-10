import datetime
import os
import argparse
import torch
from rlpyt.samplers.collections import TrajInfo

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.benchmark_dreamer_agent import BenchmarkDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.wrapper import make_wapper
from dreamer.envs.dmc import DeepMindControl
# from dreamer.envs.atari import Atari
from dreamer.envs.rlbench import RLBench
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions

from rlpyt.utils.launching.affinity import get_n_run_slots, prepend_run_slot, affinity_from_code, encode_affinity
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector

from dreamer.envs.time_limit import TimeLimit

# N means not in dreamer.experiments.scripts.train -> maybe remove these below

def build_and_train(log_dir, task="TargetReach", environments=RLBench, run_ID=0, cuda_idx=0, eval=False,               #
                    save_model='last', load_model_path=None):

    # (slot_affinity_code, log_dir, config_key)

    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = 2  # N
####

    affinity_code  = encode_affinity(
        n_cpu_core=4,
        n_gpu=0,
        hyperthread_offset=8,
        n_socket=1,
        #cpu_per_run=2,
    )
    n_run_slots = get_n_run_slots(affinity_code)
    procs = [None] * n_run_slots
    for run_slot, p in enumerate(procs):
        print("0. run_slot={}, p={}".format(run_slot, p))
        if p is None or p.poll() is not None:
            #procs[run_slot]:
            run_slot = run_slot
            affinity_code=affinity_code
            print("1. run_slot={}, affinity_code={} (best to see this once at first)".format(
                  run_slot, affinity_code))

    slot_affinity_code = prepend_run_slot(run_slot, affinity_code)
    print("slot_affinity_code", slot_affinity_code)
####
    # MOST LIKELY, THIS CAN BE HARDCODED
    affinity = affinity_from_code(slot_affinity_code)
    # affinity = affinity_from_code(slot_affinity_code)
    # config = configs[config_key]
    # variant = load_variant(log_dir)
    # config = update_config(config, variant)

    factory_method = make_wapper(
        base_class=environments,
        # wrapper_classes: list of wrapper classes in order inner-first, outer-last
        wrapper_classes=[ActionRepeat, NormalizeActions, TimeLimit],
        # list of kwargs dictionaries passed to the wrapper classes:
        wrapper_kwargs=[dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])

    # set these in env_rlbench
    environments_args = {"config": {"robot": "sawyer", "num_img_obs" : 1}}  # {task: task}}  # , "_env": ""}}
    environments_eval_args = {"config": {"robot": "sawyer", "num_img_obs": 1}}  #"task": task}

    sampler = CpuSampler(
        EnvCls=factory_method,  # AtariEnv,  # (is simpler)
        # env_kwargs=config["env"],
        CollectorCls=CpuWaitResetCollector,
        TrajInfoCls=TrajInfo,
        env_kwargs=environments_args,  # config["env"]  --> dict(game="pong, num_img_obs=1)
        eval_env_kwargs=environments_eval_args,  # N
        # **config["sampler"],  # --> dict(batch_T=20, batch_B=32, max_decorrelation_steps=1000)
        batch_T=20,  # batch_size?
        batch_B=32,  # batch_length?
        max_decorrelation_steps=1000,
        # eval_n_envs=10,  # N
        # eval_max_steps=int(10e3),  # N
        # eval_max_trajectories=5,  # N
    )
    batch_size = 50
    batch_length = 50
    algo = Dreamer(
                   optim_kwargs=dict(), **dict(),
                   # optim_kwargs=config["optim"], **config["algo"] # (probably contain the next 3 args)
                   #initial_optim_state_dict=optimizer_state_dict,  # N
                   #batch_size=batch_size,  # N
                   #batch_length=batch_length  # N
                  )
    agent = BenchmarkDreamerAgent(
                                  model_kwargs=dict(), **dict(),  # # (probably contains the next 6 args)
                                  # model_kwargs=config["model"], **config["agent"],  # # (probably contains the next 6 args)
                                  # train_noise=0.3,
                                  # eval_noise=0,
                                  # expl_type="additive_gaussian",
                                  # expl_min=None, expl_decay=None,
                                  # initial_model_state_dict=agent_state_dict
                                 )
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        # affinity=dict(cuda_idx=cuda_idx),
        affinity=affinity,  # dict(workers_cpus=[1,2,3]),  # = affinity = affinity_from_code(slot_affinity_code)
        # **config["runner"],  # --> n_steps=5e6, # log_interval_steps=1e3,
        n_steps=1e6,  # N
        # log_interval_steps=1e3,  # N
    )

    config = {"task": task}
    name = "dreamer_" + task
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
