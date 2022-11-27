from itertools import product as prod

import numpy as np


class SarsaLambda:
    def __init__(self, model, lam: float, alpha: float, gamma: float, epsilon: float, order: int, max_steps: int = 1000):
        self.model = model
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alphas = [alpha]
        self.order = order
        self.dims = len(model.getState())
        self.max_steps = max_steps
        self.num_bases = (order + 1) ** self.dims

        # set up basis function
        all_consts = list(prod(range(order + 1), repeat=self.dims))
        all_consts.remove(all_consts[0])
        self.basis = [lambda _: 0]
        for c in all_consts:
            c = np.array(c)
            self.basis.append(lambda s: np.cos(np.pi * np.dot(s, c)))
            self.alphas.append(alpha / np.linalg.norm(c))
        self.alphas = np.array(self.alphas)

        self.weights = np.zeros((self.num_bases, len(model.actions)))

    def normalize(self, s):
        S = s.copy()
        for i in range(len(S)):
            S[i] = (S[i] - self.model.min_maxes[i * 2]) /\
                   (self.model.min_maxes[i * 2 + 1] - self.model.min_maxes[i * 2])
        return np.array(S)

    def getAction(self, S) -> int:
        if np.random.uniform() <= self.epsilon:
            return np.random.choice(self.model.actions)
        return self.model.actions[np.argmax([self.value(S, A) for A in self.model.actions])]

    # estimate the value of given state and action
    def value(self, S, A: int) -> float:
        if self.model.isTerminal(S):
            return 0.0

        phi = np.array([feature(self.normalize(S)) for feature in self.basis])
        return float(np.dot(self.weights[:, 0], phi))

    def playEpisode(self) -> int:
        self.model.reset()
        S = self.model.getState()
        A = self.getAction(S)
        phi = np.array([feature(self.normalize(S)) for feature in self.basis])
        z = np.zeros((self.num_bases, len(self.model.actions)))
        Q_old = 0

        for steps in range(self.max_steps):
            A_ind = 0  # self.model.actions.index(A)
            R, S_p = self.model.update(A)
            A_p = self.getAction(S_p)
            Q = self.value(S, A)
            Q_p = self.value(S_p, A_p)
            z[:, A_ind] = self.gamma * self.lam * z[:, A_ind] +\
                          (1 - self.gamma * self.lam * np.dot(z[:, A_ind], np.multiply(self.alphas, phi))) * phi
            self.weights[:, A_ind] +=\
                (R + self.gamma * Q_p - Q_old) * np.multiply(self.alphas, z[:, A_ind]) -\
                (Q - Q_old) * np.multiply(self.alphas, phi)
            Q_old = Q_p
            phi = np.array([feature(self.normalize(S_p)) for feature in self.basis])
            A = A_p

            if self.model.isTerminal():
                return steps + 1

        return self.max_steps

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self) -> float:
        S = self.model.getState()
        return -max([self.value(S, A) for A in self.model.actions])
