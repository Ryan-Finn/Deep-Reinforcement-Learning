import numpy as np


class MountainCar:
    def __init__(self, position: float = None, velocity: float = 0.0):
        if position is None:
            position = np.random.uniform(-0.6, -0.4)
        self.x = position
        self.v = velocity
        self.actions = [-1, 0, 1]
        self.min_maxes = [-1.2, 0.5, -0.07, 0.07]  # [Min position, max position, min velocity, max velocity]

    def reset(self):
        self.x = np.random.uniform(-0.6, -0.4)
        self.v = 0.0

    def set(self, state: [float, float]):
        self.x = state[0]
        self.v = state[1]

    def update(self, action) -> [float, float]:
        self.v += 0.001 * action - 0.0025 * np.cos(np.pi * self.x)
        self.x += self.v

        if self.x < self.min_maxes[0]:
            self.x = self.min_maxes[0]
            self.v = 0

        if self.x > self.min_maxes[1]:
            self.x = self.min_maxes[1]

        if self.v < self.min_maxes[2]:
            self.v = self.min_maxes[2]

        if self.v > self.min_maxes[3]:
            self.v = self.min_maxes[3]

        return [self.x, self.v]

    def isTerminal(self, position: float = None) -> bool:
        if position is None:
            return self.x == self.min_maxes[1]
        return position == self.min_maxes[1]

    def getState(self) -> [float, float]:
        return [self.x, self.v]
