import gym
import rlbench.gym

env = gym.make('take_frame_off_hanger-state-v0', render_mode="human")
env.reset()
for _ in range(20):
    env.render()
    env.step(env.action_space.sample())
env.close()
