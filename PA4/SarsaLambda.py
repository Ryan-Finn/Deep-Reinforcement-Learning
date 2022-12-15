from itertools import product
from math import prod

import numpy as np


class SarsaLambda:
    def __init__(self, model, lam: float = 0.9, alpha: float = 0.001, gamma: float = 1.0, epsilon: float = 0.05,
                 R: float = 0.0, P: float = -1.0, order: int = 3, max_steps: int = 500, weights=None):
        self.model = model
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.R = R
        self.P = P
        self.order = order
        self.max_steps = max_steps
        self.weights = weights

        self.dims = len(model.low)
        self.num_basis = None
        self.basis = None
        self.alphas = None
        self.S = None
        self.A = None
        self.z = None
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
        if self.model.isOutOfBounds(S) or self.model.isTerminal(S):
            return self.P

        phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])
        return float(np.dot(self.weights[:, A], phi))

    def newEpisode(self):
        self.S = self.model.reset()
        self.A = self.getAction(self.S)
        self.z = np.zeros((self.num_basis, self.model.action_space.n))

    def step(self):
        phi = np.array([feature(self.model.normalize(self.S)) for feature in self.basis])

        self.z[:, self.A] += phi  # accumulating traces
        S_p, outOfBounds, terminated = self.model.step(self.A)
        R = self.R
        if outOfBounds or terminated:
            R = self.P

        delta = R - self.value(self.S, self.A)

        if outOfBounds or terminated:
            self.weights += delta * self.z * self.alphas
            return True

        A_p = self.getAction(S_p)
        delta += self.gamma * self.value(S_p, A_p)
        self.weights += delta * self.z * self.alphas
        self.z *= self.gamma * self.lam
        self.S = S_p
        self.A = A_p

        return False

    def cost_to_go(self, S) -> float:
        return -max([self.value(S, A) for A in range(self.model.action_space.n)])
