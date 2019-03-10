from math import cos

import numpy as np
from gym.envs.classic_control.mountain_car import \
    MountainCarEnv as GymMountainCarEnv


class MountainCarEnv(GymMountainCarEnv):

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001 + cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position)
        reward = 1 * (position + 1.2) if position > -0.2 else -1

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}
