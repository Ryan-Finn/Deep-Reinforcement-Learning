from math import cos, sin
from os import environ
from typing import Optional

import numpy as np
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.envs.classic_control.cartpole import CartPoleEnv

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pygame import display


class CartPole(CartPoleEnv):
    def __init__(self):
        super().__init__(render_mode="human")
        self.high = np.array(
            [
                self.x_threshold * 2,
                5.0,
                self.theta_threshold_radians * 2,
                5.0,
            ],
            dtype=np.float32,
        )
        self.low = -self.high
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        super(CartPoleEnv, self).reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.1, 0.1)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        return np.array(self.state, dtype=np.float32)

    def step(self, action: int, e: float = 0.5) -> (np.ndarray, float, bool):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = cos(theta)
        sintheta = sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x += self.tau * x_dot
            x_dot += self.tau * xacc
            theta += self.tau * theta_dot
            theta_dot += self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot += self.tau * xacc
            x += self.tau * x_dot
            theta_dot += self.tau * thetaacc
            theta += self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        if not self.isTerminal():
            return np.array(self.state, dtype=np.float32), 1.0, False
        elif self.steps_beyond_terminated is None:  # Pole just fell!
            self.steps_beyond_terminated = 0
            return np.array(self.state, dtype=np.float32), 0.0, True

        if self.steps_beyond_terminated == 0:
            logger.warn(
                "You are calling 'step()' even though this "
                "environment has already returned terminated = True. You "
                "should always call 'reset()' once you receive 'terminated = "
                "True' -- any further steps are undefined behavior."
            )
        self.steps_beyond_terminated += 1
        return np.array(self.state, dtype=np.float32), 0.0, True

    def animate(self, episode: int, step: int, max_steps: int = None, best_steps: int = None):
        super().render()
        max_str = ''
        best_str = ''

        if max_steps is not None:
            max_str = ' / ' + str(max_steps)

        if best_steps is not None:
            best_str = '    |    Best: ' + str(best_steps)

        if self.render_mode == "human":
            display.set_caption(f'    Episode: {episode}    |    Step: {step}' + max_str + best_str)

    def isTerminal(self, state: (float, float, float, float) = None) -> bool:
        if state is None:
            state = self.state
        x, x_dot, theta, theta_dot = state
        return bool(x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians
                    or theta > self.theta_threshold_radians)

    def getState(self) -> np.ndarray:
        return np.array(self.state)

    def normalize(self, state: (float, float, float, float)) -> np.ndarray:
        S = np.array(state.copy())
        for i in range(len(S)):
            S[i] = (S[i] - self.low[i]) / (self.high[i] - self.low[i])
        return S
