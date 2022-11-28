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
        self.dims = len(model.observation_space)
        self.max_steps = max_steps
        self.num_bases = (order + 1) ** self.dims

        # set up basis function
        all_consts = list(prod(range(order + 1), repeat=self.dims))
        all_consts.remove(all_consts[0])
        self.basis = [lambda _: 1]
        for const in all_consts:
            const = np.array(const)
            self.basis.append(lambda s, c=const: np.cos(np.pi * np.dot(s, c)))
            self.alphas.append(alpha / np.linalg.norm(const))
        self.alphas = np.array([self.alphas] * len(model.action_space)).T

        self.weights = np.zeros((self.num_bases, len(model.action_space)))

    def getAction(self, S) -> int:
        if np.random.uniform() <= self.epsilon:
            return np.random.choice(self.model.actions)
        return int(np.argmax([self.value(S, A) for A in range(len(self.model.actions))]))

    # estimate the value of given state and action
    def value(self, S, A: int) -> float:
        if self.model.isTerminal(S):
            return 0.0

        phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])
        return float(np.dot(self.weights[:, A], phi))

    def learnEpisode(self) -> int:
        self.model.reset()
        S = self.model.getState()
        z = np.zeros((self.num_bases, len(self.model.actions)))
        A = self.getAction(S)

        for steps in range(self.max_steps):
            phi = np.array([feature(self.model.normalize(S)) for feature in self.basis])

            z[:, A] = phi
            R, S_p = self.model.update(A)
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

    def animateEpisode(self, weights=None):
        self.weights = weights

        self.model.reset()
        self.model.render()
        S = self.model.getState()
        A = self.getAction(S)

        for steps in range(self.max_steps):
            R, S_p = self.model.update(A)
            self.model.render()

            if self.model.isTerminal():
                return steps + 1

            A_p = self.getAction(S_p)
            A = A_p

        return self.max_steps

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self) -> float:
        S = self.model.getState()
        return -max([self.value(S, A) for A in range(len(self.model.actions))])
