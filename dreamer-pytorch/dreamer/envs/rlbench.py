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
        """
        When initializing RLBench, config is provided
        """
        self.config = config
#        self._env, self._task =
        env = self._initialize()
        self._env = env
        self._task = ReachTarget  # env.get_task(self.config.get("task", ReachTarget))

    def _initialize(self):
        obs = self.config.get("obs_config", ObservationConfig(CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64)),
                                                              CameraConfig(image_size=(64, 64))))
        action_mode = self.config.get("action_mode", ActionMode(ArmActionMode.ABS_JOINT_VELOCITY))
        headless = self.config.get("headless", True)
        # RLBench inherits from Env but needs to initialize Environment
        env = Environment(action_mode, obs_config=obs, headless=headless)
        env.launch()
        # The task is hard coded here
        # methd called _initialize() should not return values
        return env  # , task

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self.config.get("size", (64, 64)),
                      dtype="uint8")

    @property
    def action_space(self):
        return FloatBox(low=-1.0, high=1.0, shape=(self._env.action_size,))

    def step(self, action):
        #task = self._env.get_task(ReachTarget)
        #task.step(action)
        task = self._env.get_task(ReachTarget)
        task.reset()
        obs, reward, done = task.step(action)
        obs = np.transpose(obs.front_rgb, (2, 0, 1))
        info = EnvInfo(None, None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        #RLBench tasks don't seem to have a reset() function
        # cf ../RLBench/rlbench/backend/task.py
        # def unload(self) -> None:
        # def cleanup_(self) -> None:
        # def clear_registerings(self) -> None:
        # descriptions, obs = self._task.reset()
        # descriptions, obs = self._task.cleanup_()
        task = self._env.get_task(ReachTarget)
        descriptions, obs = task.reset()
        obs = np.transpose(obs.front_rgb, (2, 0, 1))
        del descriptions  # Not used.
        return obs

    def render(self, *args, **kwargs):
        pass

    @property
    def horizon(self):
        raise NotImplementedError
