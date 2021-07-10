import numpy as np
import torch
import gym
import itertools as it
import sys
sys.path.append("../")
from agent.dqn_agent import DQNAgent
from agent.networks import MLP
from agent.replay_buffer import ReplayBuffer
from utils import EpisodeStats
from tensorboard_evaluation import *


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the
        Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal, env)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    """
    Evaluates agent every 'eval_cycle' episodes using run_episode(),
    to check its performance with greedy actions only, uses
    tensorboard to plot the mean episode reward.

    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), "CARTPOLE", ["episode_reward", "a_0", "a_1"])

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)

        tensorboard.write_episode_data(i, eval_dict={
                                       "episode_reward": stats.episode_reward,
                                       "a_0": stats.get_action_usage(0),
                                       "a_1": stats.get_action_usage(1)})

        if i % eval_cycle == 0:
            for j in range(num_eval_episodes):
                run_episode(env, agent, deterministic=True,
                            do_training=False)

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False

    epsilon = 0.1
    gamma = 0.95            # Q-learning discount factor
    lr = 1e-4               # NN optimizer learning rate
    hidden_dim = 400        # NN hidden layer size
    batch_size = 64         # Q-learning batch size
    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes
    tau = 0.01
    history_length = 1e5

    env = gym.make("CartPole-v0").unwrapped

    num_episodes = 100

    state_dim = 4
    num_actions = 2
    action_dim = num_actions

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    Q_target = MLP(state_dim, action_dim, hidden_dim=400)
    Q = MLP(state_dim, action_dim, hidden_dim=400)
    if use_cuda:
        model.cuda()
    memory = ReplayBuffer(history_length)
    optimizer = torch.optim.Adam(Q.parameters(), lr)
    steps_done = 0
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(Q, Q_target, num_actions, gamma, batch_size,
                     epsilon, tau, lr, history_length)
    """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Q: Action-Value function estimator (Neural Network)
        Q_target: Slowly updated target network to calculate the targets.
        num_actions: Number of actions of the environment.
        gamma: discount factor of future rewards.
        batch_size: Number of samples per batch.
        tau: indicates the speed of adjustment of the slowly updated target network.
        epsilon: Chance to sample a random action. Float between 0 and 1.
        lr: learning rate of the optimizer"""

    # 3. train DQN agent with train_online(...)
    train_online(env, agent, num_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard")
