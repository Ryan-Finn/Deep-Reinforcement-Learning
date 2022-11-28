from itertools import product as prod
import numpy as np


class SarsaLambda:
    def __init__(self, model, lam: float, alpha: float, gamma: float, epsilon: float, order: int,
                 max_steps: int = 1000, animate: bool = False):
        if animate:
            model.render_mode = 'human'
        self.model = model
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alphas = [alpha]
        self.order = order
        self.dims = len(model.low)
        self.max_steps = max_steps
        self.num_basis = (order + 1) ** self.dims

        # Set up Fourier basis functions and alpha values
        all_consts = list(prod(range(order + 1), repeat=self.dims))
        all_consts.remove(all_consts[0])
        self.basis = [lambda _: 1]
        for i in range(self.num_basis - 1):
            const = np.array(all_consts[i])
            self.basis.append(lambda s, c=const: np.cos(np.pi * np.dot(s, c)))
            self.alphas.append(alpha / np.linalg.norm(const))  # a_i = a / ||c||
        self.alphas = np.array([self.alphas] * model.action_space.n).T

        self.weights = np.zeros((self.num_basis, model.action_space.n))

    def getAction(self, S) -> int:
        if np.random.uniform() <= self.epsilon:
            return self.model.action_space.sample()
        return int(np.argmax([self.value(S, A) for A in range(self.model.action_space.n)]))

    # estimate the value of given state and action
    def value(self, S, A: int) -> float:
        if self.model.isTerminal(S):
            return 0.0

        phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])
        return float(np.dot(self.weights[:, A], phi))

    def learnEpisode(self) -> int:
        self.model.reset()
        S = self.model.getState()
        z = np.zeros((self.num_basis, self.model.action_space.n))
        A = self.getAction(S)

        for steps in range(self.max_steps):
            phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])

            z[:, A] = phi
            R, S_p = self.model.step(A)
            delta = R - self.value(S, A)

            if self.model.isTerminal():
                self.weights += delta * z * self.alphas
                return steps + 1

            A_p = self.getAction(S_p)
            delta += self.gamma * self.value(S_p, A_p)
            self.weights += delta * z * self.alphas
            z *= self.gamma * self.lam
            S = S_p
            A = A_p

        return self.max_steps

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self) -> float:
        S = self.model.getState()
        return -max([self.value(S, A) for A in range(self.model.action_space.n)])
