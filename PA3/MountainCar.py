import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv


class MountainCar(MountainCarEnv):
    # def reset():
    #     self.state, _ = self.env.reset()

    def set(self, state: [float, float]):
        self.state, _ = self.env.reset(options={'low': state[0], 'high': state[0]})

    def update(self, action: int) -> (int, [float, float]):
        self.env.step(action)
        self.v += 0.001 * (action - 1) - 0.0025 * np.cos(3 * self.x)

        if self.v < self.min_maxes[2]:
            self.v = self.min_maxes[2]

        if self.v > self.min_maxes[3]:
            self.v = self.min_maxes[3]

        self.x += self.v

        if self.x < self.min_maxes[0]:
            self.x = self.min_maxes[0]
            self.v = 0

        if self.x > self.min_maxes[1]:
            self.x = self.min_maxes[1]
            return 0, [self.x, 0]

        return -1, [self.x, self.v]

    def isTerminal(self, state: list[float, float] = None) -> bool:
        if state is None:
            return self.x == self.min_maxes[1]
        return state[0] == self.min_maxes[1]

    def getState(self) -> [float, float]:
        return [self.x, self.v]
