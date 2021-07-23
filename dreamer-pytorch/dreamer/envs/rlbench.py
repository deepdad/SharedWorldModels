import threading

import numpy as np
from pyrep.const import RenderMode
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import CloseDrawer

from dreamer.envs.env import EnvInfo


class RLBench(Env):

    def __init__(self, config):
        self.config = config
        self._env, self._task = self._initialize()

    def _initialize(self):
        # None actually produces the same
        self.obs_sz = (64, 64)
#        obs = self.config.get("obs_config", ObservationConfig(CameraConfig(image_size=self.obs_sz),
#                                                              CameraConfig(image_size=self.obs_sz),
#                                                              CameraConfig(image_size=self.obs_sz),
#                                                              CameraConfig(image_size=self.obs_sz),
#                                                              CameraConfig(image_size=self.obs_sz)))
        cam_config = CameraConfig(image_size=self.obs_sz, render_mode=RenderMode.OPENGL3)
        obs_config = ObservationConfig(wrist_camera=cam_config)
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)
        obs_config.wrist_camera.set_all(True)
        obs_config.front_camera.set_all(False)  # note: TODO: test whether shoulder camera works better
        self.action_mode = self.config.get("action_mode",
                                      ActionMode(ArmActionMode.ABS_JOINT_VELOCITY))
        headless = self.config.get("headless", True)
        #headless = self.config.get("headless", False)
        env = Environment(self.action_mode, obs_config=obs_config, headless=headless)
        env.launch()
        self.action_space = self.config.get("action_space",
                                            FloatBox(low=-1.0, high=1.0, shape=(env.action_size,)))
        task = env.get_task(self.config.get("task", CloseDrawer))
        return env, task

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self.config.get("size", self.obs_sz),
                      dtype="uint8")

    def action_space(self):
        return self.action_space

    def step(self, action):
        obs, reward, done = self._task.step(action)
        obs = np.transpose(obs.wrist_rgb, (2, 0, 1))
        info = EnvInfo(None, None, done, self.action_mode.arm.value)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        descriptions, obs = self._task.reset()
        obs = np.transpose(obs.wrist_rgb, (2, 0, 1))
        del descriptions  # Not used.
        return obs

    def render(self, *args, **kwargs):
        pass

    def shutdown(self):
        self._env.shutdown()
        print("done shutdown")

    @property
    def horizon(self):
        raise NotImplementedError


