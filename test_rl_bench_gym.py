import time

import rlbench.gym as gym
from rlbench.tasks import ReachTarget

env = gym.RLBenchEnv(ReachTarget, observation_mode="vision", render_mode='rgb_array')

training_steps = 1000
episode_length = 40
before = time.time()
for i in range(training_steps):
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    print(i)
print('Done')
after = time.time()
print(after-before)
env.close()