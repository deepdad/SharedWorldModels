import threading

import numpy as np
from pyrep.const import RenderMode
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import SmallTargetReachNoDistractors
from rlbench.tasks import LargeTargetNoDistractorsDt200

from dreamer.envs.env import EnvInfo


class RLBench(Env):

    def __init__(self, config):
        self.config = config
        self._env, self._task = self._initialize()

    def _initialize(self):
        self.obs_sz = (64, 64)
        cam_config = CameraConfig(image_size=self.obs_sz , render_mode=RenderMode.OPENGL3)
        obs_config = ObservationConfig(wrist_camera=cam_config)
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)
        obs_config.wrist_camera.set_all(True)
        obs_config.front_camera.set_all(False)  # note: TODO: test whether shoulder camera works better
        print("\nOBS1", vars(obs_config))
        for ok, ov in vars(obs_config).items():
            if "camera" in ok and "matrix" not in ok:
                print(ok, vars(ov))
        action_mode = self.config.get("action_mode",
                                      ActionMode(ArmActionMode.ABS_JOINT_VELOCITY))
        headless = self.config.get("headless", False)
        env = Environment(action_mode, obs_config=obs_config, headless=headless)
        env.launch()
        task = env.get_task(self.config.get("task", ReachTarget))
        return env, task

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self.config.get("size", self.obs_sz),
                      dtype="uint8")

    @property
    def action_space(self):
        print("Does the action space {} make sense?".format(FloatBox(low=-1.0,
                        high=1.0,
                        shape=(self._env.action_size,))))
        return FloatBox(low=-1.0,
                        high=1.0,
                        shape=(self._env.action_size,))

    def step(self, action):
        obs, reward, done = self._task.step(action)
        #obs = np.transpose(obs.front_rgb, (2, 0, 1))
        obs = np.transpose(obs.wrist_rgb, (2, 0, 1))
        info = EnvInfo(None, None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        descriptions, obs = self._task.reset()
        # obs = np.transpose(obs.front_rgb, (2, 0, 1))
        obs = np.transpose(obs.wrist_rgb, (2, 0, 1))
        del descriptions  # Not used.
        return obs

    def render(self, *args, **kwargs):
        pass

    @property
    def horizon(self):
        raise NotImplementedError
