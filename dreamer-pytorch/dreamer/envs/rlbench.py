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
        # None actually produces the same
        obs = self.config.get("obs_config", ObservationConfig(CameraConfig(image_size=(2, 2)),
                                                              CameraConfig(image_size=(2, 2)),
                                                              CameraConfig(image_size=(2, 2)),
                                                              CameraConfig(image_size=(2, 2)),
                                                              CameraConfig(image_size=(64, 64))))
        print("OBS0", vars(obs))
#        obs.left_shoulder_camera.set_all(False)
#        obs.right_shoulder_camera.set_all(False)
#        obs.overhead_camera.set_all(False)
#        obs.wrist_camera.set_all(False)

#        for ok, ov in vars(obs).items():
#            if "camera" in ok and "matrix" not in ok:
#                print(ok, vars(ov))
#                obs.
#            if "front_camera"
#        obs.set_all(False)
#        print("\nOBS1", vars(obs))
#        for ok, ov in vars(obs).items():
#            if "camera" in ok and "matrix" not in ok:
#                print(ok, vars(ov))
        obs.front_camera.set_all(True)
#        print("\nOBS2", vars(obs))
#        for ok, ov in vars(obs).items():
#            if "camera" in ok and "matrix" not in ok:
#                print(ok, vars(ov))
        action_mode = self.config.get("action_mode", ActionMode(ArmActionMode.ABS_JOINT_VELOCITY))
        headless = self.config.get("headless", True)
        env = Environment(action_mode, obs_config=obs, headless=headless,
                          robot_configuration="panda" #panda', 'jaco', 'mico', 'sawyer', 'ur5'
                         )
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
        print("RLBENCH RESET 1", self._task, vars(self._task), vars(vars(self._task)['_scene']))
        descriptions, obs = self._task.reset()
        print("RLBENCH RESET 2")
        obs = np.transpose(obs.front_rgb, (2, 0, 1))
        del descriptions  # Not used.
        print("RLBENCH RESET 3")
        return obs

    def render(self, *args, **kwargs):
        pass

    @property
    def horizon(self):
        raise NotImplementedError
