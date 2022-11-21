import numpy as np

from Tiles import IHT, tiles


class SarsaLambda:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, actions, min_maxes, discount, num_of_tilings=8, max_size=2048):
        self.actions = actions
        self.discount = discount
        self.num_of_tilings = num_of_tilings
        self.max_size = max_size

        # set up bases function
        self.bases = []
        for n in range(3 + 1):
            c = []
            for d in range(2):
                c.append(d)
            self.bases.append(lambda s: np.cos(np.pi * c * s))

        self.alpha, self.lam = None, None

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(3 + 1)

        # trace for each tile
        self.trace = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (min_maxes[1] - min_maxes[0])
        self.velocity_scale = self.num_of_tilings / (min_maxes[3] - min_maxes[2])

    def setEvaluator(self, alpha, lam):
        self.alpha = alpha  # / self.num_of_tilings
        self.lam = lam
        self.hash_table = IHT(self.max_size)
        self.weights = np.zeros(3 + 1)
        self.trace = np.zeros(self.max_size)
        return self

    # get indices of active tiles for given state and action
    def get_active_tiles(self, state, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However, positionScale * position_min is a constant, so it's ok to ignore it.
        return tiles(self.hash_table, self.num_of_tilings,
                     [self.position_scale * state[0], self.velocity_scale * state[1]], [action])

    # estimate the value of given state and action
    def value(self, model, action):
        if model.isTerminal():
            return 0.0

        # return np.sum(self.weights[self.get_active_tiles(model.getState(), action)])

        state = model.getState()
        state /= float(self.max_size)
        feature = np.asarray([func(state) for func in self.bases])

        return np.dot(self.weights, feature)

    # learn with given state, action and target
    def learn(self, state, action, delta):
        state /= float(self.max_size)
        derivative_value = np.asarray([func(state) for func in self.bases])
        # delta = self.alpha * (reward - self.value(model, action))
        self.weights += self.alpha * delta * derivative_value

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
    def cost_to_go(self, model):
        costs = []
        for action in self.actions:
            costs.append(self.value(model, action))
        return -np.max(costs)
