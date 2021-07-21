from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachTarget(Task):

    def init_task(self) -> None:
        self.printreward = 0.0
        self.target = Shape('target')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        self.printreward = 0.0
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        self._initial_distance = self._armtip_target_difference()
        self._prev_distance = self._initial_distance

        return ['reach the red target',
                'touch the red ball with the panda gripper',
                'reach the red sphere']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())

    def is_static_workspace(self) -> bool:
        return True

    def reward(self):
        distance = self._armtip_target_difference()
        reward = (self._prev_distance - distance) / self._initial_distance
        self._prev_distance = distance
        self.printreward += reward
        return reward

    def _armtip_target_difference(self):
        armtip_position = self.robot.arm.get_tip().get_position()
        target_position = self.target.get_position()
        return np.linalg.norm(np.array(armtip_position) - np.array(target_position))
