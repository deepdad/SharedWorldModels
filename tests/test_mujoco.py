# Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
# Setting FetchReach-v1 as the environment
env = gym.make('FetchReach-v1')
# Set an initial state
env.reset()
# Rendering our instance 300 times
for _ in range(100):
  # render the environment
  env.render()
  # Take a random action from the action space
  env.step(env.action_space.sample())
env.close()
