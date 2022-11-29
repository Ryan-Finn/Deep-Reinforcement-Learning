from math import cos
from os import environ
from typing import Optional

import numpy as np
from gym.envs.classic_control import utils
from gym.envs.classic_control.mountain_car import MountainCarEnv

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pygame import display


class MountainCar(MountainCarEnv):
    def __init__(self):
        super().__init__(render_mode="human")
        self.state = None

    def set(self, state: (float, float)):
        self.state = state

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        super(MountainCarEnv, self).reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
        return np.array(self.state, dtype=np.float32)

    def step(self, action: int, e: float = 0.5) -> (np.ndarray, float, bool):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state

        velocity += (action - 1) * self.force + cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = -e * velocity

        self.state = (position, velocity)

        if self.isTerminal():
            return np.array(self.state, dtype=np.float32), 0.0, True
        return np.array(self.state, dtype=np.float32), -1.0, False

    def animate(self, episode: int, step: int, max_steps: int):
        super().render()
        if self.render_mode == "human":
            display.set_caption(f'    Episode: {episode}    |    Step: {step} / {max_steps}')

    def isTerminal(self, state: (float, float) = None) -> bool:
        if state is None:
            state = self.state
        return state[0] >= self.goal_position  # and state[1] >= self.goal_velocity

    def getState(self) -> np.ndarray:
        return np.array(self.state)

    def normalize(self, state) -> np.ndarray:
        S = np.array(state.copy())
        for i in range(len(S)):
            S[i] = (S[i] - self.low[i]) / (self.high[i] - self.low[i])
        return S
