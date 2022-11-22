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

        self.weights = np.zeros(self.num_bases)
        self.trace = np.zeros(int(np.ceil(np.log2(0.05) / np.log2(lam)) + 1))
        self.trace[0] = 1
        for i in range(1, len(self.trace)):
            self.trace[i] = self.trace[i - 1] * lam
        print(len(self.trace), self.trace)

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
            features.append(self.alphas[i] * self.bases(state, self.consts[i]))
        features = np.array(features)

        self.model.set(prev)
        return np.dot(self.weights, features)

    # learn with given state and delta
    def learn(self, state, _, delta):
        # Normalize the state | x = (x - x_min) / (x_max - x_min)
        for i in range(len(state)):
            state[i] = (state[i] - self.model.min_maxes[i * 2]) /\
                       (self.model.min_maxes[i * 2 + 1] - self.model.min_maxes[i * 2])
        state = np.array(state)

        features = []
        for i in range(self.num_bases):
            features.append(self.alphas[i] * self.bases(state, self.consts[i]))
        features = np.array(features)

        self.trace *= self.lam
        self.trace = np.roll(self.trace, 1)
        self.trace[0] = np.dot(self.weights, features)
        print(self.trace, np.dot(self.weights, features))

        # delta -= np.sum(self.weights)

        sum_trace = np.sum(self.trace)

        self.weights += sum_trace * delta

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
