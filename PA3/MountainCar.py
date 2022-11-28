import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv


class MountainCar(MountainCarEnv):
    def set(self, state: [float, float]):
        super().state = np.array(state)
        if super().render_mode == "human":
            self.render()

    def step(self, action: int) -> (int, np.ndarray):
        state, reward, terminated, truncated, _ = super().step(action)

        if terminated or truncated:
            return 0, state
        return reward, state

    def isTerminal(self, state: list[float, float] = None) -> bool:
        if state is None:
            return bool(super().state[0] >= super().goal_position and super().state[1] >= super().goal_velocity)
        return bool(state[0] >= super().goal_position and state[1] >= super().goal_velocity)

    def getState(self) -> np.ndarray:
        return super().state

    def normalize(self, state):
        S = state.copy()
        for i in range(len(S)):
            S[i] = (S[i] - super().low[i]) / (super().high[i] - super().low[i])
        return S
