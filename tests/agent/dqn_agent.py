import tensorflow as tf
import numpy as np
import torch
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, memory, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # setup networks
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = memory  # ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, current_state, action, next_state, reward, terminal, env):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        x_next = torch.from_numpy(np.concatenate((next_state, next_state-current_state))).float()
        self.replay_buffer.add_transition(current_state, action, next_state, reward, terminal)

        # print("growing?", self.replay_buffer.size())

        if terminal:
            current_state = env.reset()
        else:
            current_state = next_state
        # 2. sample next batch and perform batch update:
        # get one minibatch:
        if len(self.replay_buffer) < self.batch_size:
            return
        self.Q.train()
        xs, actions, next_xs, rs, terminals = self.replay_buffer.next_batch(self.batch_size)
        xs = torch.Tensor(xs)
        next_xs = torch.Tensor(next_xs)
        rs = np.array(rs)
        final_state_ids = np.nonzero(terminals)
        rs = torch.from_numpy(rs).float()

        # 2.1 compute td targets and loss
        # td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        next_xs = next_xs.view(-1, 9216)
        xs = xs.view(-1, 9216)
        with torch.no_grad():
            Q_next_ = self.Q_target(next_xs)
        self.optimizer.zero_grad()
        Q_value = self.Q(xs)

        Q_next_max, Q_next_argmax = torch.max(Q_next_, 1)  # maybe 1 should be action here?
        V_next = Q_next_max
        V_next[final_state_ids] = 0
        td_target = rs + self.gamma*V_next

        actions = torch.tensor(actions).view(-1, 1)
        Q_relevant = torch.gather(Q_value, 1, actions).squeeze()
        loss = self.loss_function(Q_relevant, td_target)

        # 2.2 update the Q network
        loss.backward()
        self.optimizer.step()

        # 2.3 call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the
        Q-function approximator and epsilon (probability to select a
        random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute
              the argmax action (False in training, True in
              evaluation)
        Returns:
            action id
        """
        x = torch.from_numpy(state).float()
        x = x
        if np.random.rand() < self.epsilon:
            action_id = np.random.randint(2)
        else:
            self.Q.eval()
            # print("ACTING with ", x.view(1, -1).shape)
            q = self.Q(x.view(1, -1))
            action_id = np.argmax(q.detach().cpu().numpy())
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
