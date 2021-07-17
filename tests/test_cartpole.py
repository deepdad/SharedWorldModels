import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import *
import numpy as np
from agent.replay_buffer import ReplayBuffer

np.random.seed(0)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True if torch.cuda.is_available() else False

    # hyper parameters
    epsilon = 0.1
    gamma = 0.95            # Q-learning discount factor
    lr = 1e-4               # NN optimizer learning rate
    hidden_dim = 400        # NN hidden layer size
    batch_size = 64         # Q-learning batch size
    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes
    tau = 0.01
    history_length = 1e5    # - 1e6
    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100
    # consecutive trials.

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

    env = gym.make("CartPole-v0").unwrapped

    agent = DQNAgent(Q, Q_target, num_actions, gamma, batch_size,
                     epsilon, tau, lr, history_length)

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
