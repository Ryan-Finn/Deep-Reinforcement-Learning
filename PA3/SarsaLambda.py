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

        self.alpha, self.lam, self.trace_updater = None, None, None
        self.traceTypes = [
            self.dutch_trace,
            self.replacing_trace,
            self.replacing_trace_with_clearing,
            self.accumulating_trace
        ]

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(self.max_size)

        # trace for each tile
        self.trace = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (min_maxes[1] - min_maxes[0])
        self.velocity_scale = self.num_of_tilings / (min_maxes[3] - min_maxes[2])

    def setEvaluator(self, alpha, lam, trace=None):
        self.alpha = alpha / self.num_of_tilings
        self.lam = lam

        if trace is None:
            self.trace_updater = self.replacing_trace
        else:
            self.trace_updater = trace

        self.hash_table = IHT(self.max_size)
        self.weights = np.zeros(self.max_size)
        self.trace = np.zeros(self.max_size)

        return self

    # get indices of active tiles for given state and action
    def get_active_tiles(self, state, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However, positionScale * position_min is a constant, so it's ok to ignore it.
        return tiles(self.hash_table, self.num_of_tilings,
                     [self.position_scale * state[0], self.velocity_scale * state[1]], [action])

    # estimate the value of given state and action
    def value(self, mountain_car, action):
        if mountain_car.isTerminal():
            return 0.0
        return np.sum(self.weights[self.get_active_tiles(mountain_car.getState(), action)])

    # learn with given state, action and target
    def learn(self, state, action, target):
        active_tiles = self.get_active_tiles(state, action)
        delta = target - np.sum(self.weights[active_tiles])

        if self.trace_updater != self.replacing_trace_with_clearing:
            self.trace_updater(active_tiles)
        else:
            clearing_tiles = []
            for act in self.actions:
                if act != action:
                    clearing_tiles.extend(self.get_active_tiles(state, act))
            self.trace_updater(active_tiles, clearing_tiles)

        self.weights += self.alpha * delta * self.trace

    # accumulating trace update rule
    # @trace: old trace (will be modified)
    # @activeTiles: current active tile indices
    # @lam: lambda
    # @return: new trace for convenience
    def accumulating_trace(self, active_tiles, _=None):
        self.trace *= self.lam * self.discount
        self.trace[active_tiles] += 1
        return self.trace

    # replacing trace update rule
    # @trace: old trace (will be modified)
    # @activeTiles: current active tile indices
    # @lam: lambda
    # @return: new trace for convenience
    def replacing_trace(self, activeTiles):
        active = np.in1d(np.arange(len(self.trace)), activeTiles)
        self.trace[active] = 1
        self.trace[~active] *= self.lam * self.discount
        return self.trace

    # replacing trace update rule, 'clearing' means set all tiles corresponding to non-selected actions to 0
    # @trace: old trace (will be modified)
    # @activeTiles: current active tile indices
    # @lam: lambda
    # @clearingTiles: tiles to be cleared
    # @return: new trace for convenience
    def replacing_trace_with_clearing(self, active_tiles, clearing_tiles):
        active = np.in1d(np.arange(len(self.trace)), active_tiles)
        self.trace[~active] *= self.lam * self.discount
        self.trace[clearing_tiles] = 0
        self.trace[active] = 1
        return self.trace

    # Dutch trace update rule
    # @trace: old trace (will be modified)
    # @activeTiles: current active tile indices
    # @lam: lambda
    # @alpha: step size for all tiles
    # @return: new trace for convenience
    def dutch_trace(self, active_tiles):
        coef = 1 - self.alpha * self.discount * self.lam * np.sum(self.trace[active_tiles])
        self.trace *= self.discount * self.lam
        self.trace[active_tiles] += coef
        return self.trace
