import numpy as np
from itertools import product as prod
from random import sample


class SarsaLambda:
    def __init__(self, lam, alpha, model, order=3, dims=2):
        self.lam = lam
        self.alphas = [alpha]
        self.model = model
        self.order = order
        self.dims = dims

        # set up bases function
        self.num_bases = (order + 1) ** dims
        self.bases = lambda s, c: np.cos(np.pi * np.dot(s, c))
        all_consts = list(prod(range(order + 1), repeat=dims))
        self.consts = [np.array(all_consts[0])]
        all_consts.remove(all_consts[0])
        for const in sample(all_consts, self.num_bases - 1):
            const = np.array(const)
            self.consts.append(const)
            self.alphas.append(alpha / np.linalg.norm(const))

        # weight for each tile
        self.weights = np.zeros(self.num_bases)

        # trace for each tile
        self.trace = np.zeros(self.num_bases)

    # estimate the value of given state and action
    def value(self, action):
        if self.model.isTerminal():
            return 0.0

        prev = self.model.getState()

        self.model.update(action)
        state = self.model.getState()
        for i in range(len(state)):
            state[i] = (state[i] - self.model.min_maxes[i * 2]) /\
                       (self.model.min_maxes[i * 2 + 1] - self.model.min_maxes[i * 2])
        state = np.array(state)

        features = []
        for i in range(self.num_bases):
            features.append(self.bases(state, self.consts[i]))
        features = np.array(features)

        self.model.set(prev)
        # print(self.weights, features)
        return np.dot(self.weights, features)

    # learn with given state and delta
    def learn(self, state, delta):
        for i in range(len(state)):
            state[i] = (state[i] - self.model.min_maxes[i * 2]) /\
                       (self.model.min_maxes[i * 2 + 1] - self.model.min_maxes[i * 2])
        state = np.array(state)

        for i in range(self.num_bases):
            delta -= self.weights[i] * self.bases(state, self.consts[i])

        for i in range(self.num_bases):
            self.weights[i] += self.lam * self.alphas[i] * delta

        # active_tiles = self.get_active_tiles(state, action)
        # delta = target - np.sum(self.weights[active_tiles])
        #
        # # Replacing Trace
        # active = np.in1d(np.arange(len(self.trace)), active_tiles)
        # self.trace[active] = 1
        # self.trace[~active] *= self.lam * self.discount
        #
        # self.weights += self.alpha * delta * self.trace

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self):
        costs = []
        for action in self.model.actions:
            costs.append(self.value(action))
        return -np.max(costs)