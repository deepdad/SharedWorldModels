import numpy as np
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget

from dreamer.envs.env import EnvInfo


class RLBench(Env):

    def __init__(self, config):
        self.config = config
        self._env, self._task = self._initialize()

    def _initialize(self):
        obs = self.config.get("obs_config", ObservationConfig(CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64))))
        action_mode = self.config.get("action_mode", ActionMode(ArmActionMode.ABS_JOINT_VELOCITY))
        headless = self.config.get("headless", True)
        env = Environment(action_mode, obs_config=obs, headless=headless)
        env.launch()
        task = env.get_task(self.config.get("task", ReachTarget))
        return env, task

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self.config.get("size", (64, 64)),
                      dtype="uint8")

    @property
    def action_space(self):
        return FloatBox(low=-1.0, high=1.0, shape=(self._env.action_size,))

    def step(self, action):
        obs, reward, done = self._task.step(action)
        obs = np.transpose(obs.front_rgb, (2, 0, 1))
        info = EnvInfo(None, None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        descriptions, obs = self._task.reset()
        obs = np.transpose(obs.front_rgb, (2, 0, 1))
        del descriptions  # Not used.
        return obs

    def render(self, *args, **kwargs):
        pass

    @property
    def horizon(self):
        raise NotImplementedError