from itertools import product
from math import prod

import numpy as np


class SarsaLambda:
    def __init__(self, model, lam: float = 0.9, alpha: float = 0.001, gamma: float = 1.0, epsilon: float = 0.05,
                 order: int = 3, max_steps: int = 500, weights=None):
        self.model = model
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.order = order
        self.max_steps = max_steps
        self.weights = weights

        self.dims = len(model.low)
        self.num_basis = None
        self.basis = None
        self.alphas = None
        self.best_steps = 0

        self.setFourier(order)
        # self.setPolynomial(order)

    def setFourier(self, order):
        self.alphas = [self.alpha]
        self.num_basis = (order + 1) ** self.dims

        all_consts = list(product(range(order + 1), repeat=self.dims))
        all_consts.remove(all_consts[0])
        self.basis = [lambda _: 1]

        for i in range(self.num_basis - 1):
            const = np.array(all_consts[i])
            self.basis.append(lambda s, c=const: np.cos(np.pi * np.dot(s, c)))
            self.alphas.append(self.alpha / np.linalg.norm(const))  # a_i = a / ||c_i||

        self.alphas = np.array([self.alphas] * self.model.action_space.n).T

        if self.weights is None:
            self.weights = np.zeros((self.num_basis, self.model.action_space.n))

    def setPolynomial(self, order):
        self.alphas = [self.alpha]
        self.num_basis = (order + 1) ** self.dims

        all_consts = list(product(range(order + 1), repeat=self.dims))
        all_consts.remove(all_consts[0])
        self.basis = [lambda _: 1]

        for i in range(self.num_basis - 1):
            const = np.array(all_consts[i])
            self.basis.append(lambda s, c=const: prod(np.power(s, const)))
            self.alphas.append(self.alpha / np.linalg.norm(const))  # a_i = a / ||c_i||

        self.alphas = np.array([self.alphas] * self.model.action_space.n).T

        if self.weights is None:
            self.weights = np.zeros((self.num_basis, self.model.action_space.n))

    def getAction(self, S) -> int:
        if np.random.uniform() <= self.epsilon:
            return self.model.action_space.sample()
        return int(np.argmax([self.value(S, A) for A in range(self.model.action_space.n)]))

    def value(self, S, A: int) -> float:
        if self.model.isTerminal(S):
            return 0.0

        phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])
        return float(np.dot(self.weights[:, A], phi))

    def learnEpisode(self, episode: int = 0, animate: bool = False) -> int:
        self.model.reset()
        if animate:
            self.model.animate(episode, 0, self.max_steps)
        S = self.model.getState()
        A = self.getAction(S)
        z = np.zeros((self.num_basis, self.model.action_space.n))

        for steps in range(self.max_steps):
            phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])

            z[:, A] += phi  # accumulating traces
            S_p, R, terminated = self.model.step(A)
            if animate:
                self.model.animate(episode, steps + 1, self.max_steps)
            delta = R - self.value(S, A)

            if terminated:
                self.weights += delta * z * self.alphas
                return steps + 1

            A_p = self.getAction(S_p)
            delta += self.gamma * self.value(S_p, A_p)
            self.weights += delta * z * self.alphas
            z *= self.gamma * self.lam
            S = S_p
            A = A_p

        return self.max_steps

    def playEpisode(self, episode: int = 0) -> int:
        self.model.reset()
        self.model.animate(episode, 0, best_steps=self.best_steps)
        S = self.model.getState()
        A = self.getAction(S)

        steps = 1
        while True:
            if steps > self.best_steps:
                self.best_steps = steps

            S_p, _, terminated = self.model.step(A)
            self.model.animate(episode, steps, best_steps=self.best_steps)

            if terminated:
                return steps

            A = self.getAction(S_p)
            steps += 1

    def cost_to_go(self, S) -> float:
        return -max([self.value(S, A) for A in range(self.model.action_space.n)])
