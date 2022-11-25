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
        self.basis = [lambda s: np.cos(np.pi * np.dot(s, np.array(all_consts[0])))]
        all_consts.remove(all_consts[0])

        for c in all_consts:
            c = np.array(c)
            self.basis.append(lambda s: np.cos(np.pi * np.dot(s, c)))
            self.alphas.append(alpha / np.linalg.norm(c))
        self.alphas = np.array(self.alphas)

        self.weights = np.zeros((self.num_bases, len(model.actions)))

    def normalize(self, s):
        S = s.copy()
        for i in range(len(S)):
            S[i] = (S[i] - self.model.min_maxes[i * 2]) / \
                   (self.model.min_maxes[i * 2 + 1] - self.model.min_maxes[i * 2])
        return np.array(S)

    def getAction(self, phi) -> int:
        if np.random.uniform() <= self.epsilon:
            return np.random.choice(self.model.actions)
        return self.model.actions[np.argmax([self.value(phi, A) for A in self.model.actions])]

    # estimate the value of given state and action
    def value(self, phi, A: int) -> float:
        return float(np.dot(self.weights[:, self.model.actions.index(A)], phi))

    def playEpisode(self) -> int:
        self.model.reset()
        S = self.model.getState()
        phi = np.array([feature(self.normalize(S)) for feature in self.basis])
        A = self.getAction(phi)
        z = np.zeros((self.num_bases, len(self.model.actions)))

        for steps in range(self.max_steps):
            R, S_p = self.model.update(A)
            phi_p = np.array([feature(self.normalize(S_p)) for feature in self.basis])
            A_p = self.getAction(phi_p)
            A_ind = self.model.actions.index(A_p)
            Q = np.dot(self.weights[:, self.model.actions.index(A)], phi)
            Q_p = np.dot(self.weights[:, A_ind], phi_p)

            delta = R - Q
            if not self.model.isTerminal():
                delta += self.gamma * Q_p

            z[:, A_ind] = phi

            for a in range(len(self.model.actions)):
                self.weights[:, a] += delta * np.multiply(self.alphas, z[:, a])

            z *= self.gamma * self.lam

            phi = phi_p
            A = A_p

            if self.model.isTerminal():
                return steps + 1

        return self.max_steps

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self) -> float:
        S = self.model.getState()
        phi = np.array([feature(self.normalize(S)) for feature in self.basis])
        return -max([self.value(phi, A) for A in self.model.actions])
