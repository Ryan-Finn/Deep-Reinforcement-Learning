import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv


class MountainCar(MountainCarEnv):
    def __init__(self):
        super().__init__()

    def set(self, state: [float, float]):
        self.state = np.array(state)
        if self.render_mode == "human":
            super().render()

    def step(self, action: int) -> (int, np.ndarray):
        state, reward, terminated, truncated, _ = super().step(action)

        if terminated or truncated:
            return 0, state
        return reward, state

    def isTerminal(self, state: list[float, float] = None) -> bool:
        if state is None:
            state = self.state
        return state[0] >= self.goal_position and state[1] >= self.goal_velocity

    def getState(self) -> np.ndarray:
        return self.state

    def normalize(self, state):
        S = state.copy()
        for i in range(len(S)):
            S[i] = (S[i] - self.low[i]) / (self.high[i] - self.low[i])
        return S
