import numpy as np


class MountainCar:
    def __init__(self, position: float = np.random.uniform(-0.6, -0.4), velocity: float = 0.0, terminal: float = 0.5):
        self.x = position
        self.v = velocity
        self.t = terminal

    def reset(self):
        self.x = np.random.uniform(-0.6, -0.4)
        self.v = 0.0

    def set(self, position: float, velocity: float):
        self.x = position
        self.v = velocity

    def update(self, action) -> [float, float]:
        self.v += 0.001 * action - 0.0025 * np.cos(3 * self.x)
        self.x += self.v

        if self.v < -0.07:
            self.v = -0.07

        if self.v > 0.07:
            self.v = 0.07

        if self.x < -1.2:
            self.x = -1.2
            self.v = 0

        if self.x > self.t:
            self.x = self.t

        return [self.x, self.v]

    def isTerminal(self, position: float = None) -> bool:
        if position is None:
            return self.x == self.t
        return position == self.t

    def getState(self) -> [float, float]:
        return [self.x, self.v]
