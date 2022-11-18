#######################################################################
# Copyright (C)                                                       #
# 2017-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import floor
from tqdm import tqdm
import sim

matplotlib.use('Agg')


#######################################################################
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
#
# Tile coding starts
class IHT:
    """Structure to handle collisions"""

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0:
                print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    if m is None:
        return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    ts = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        ts.append(hash_coords(coords, iht_or_size, read_only))
    return ts


# Tile coding ends
#######################################################################

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# discount is always 1.0 in these experiments
DISCOUNT = 1.0

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# maximum steps per episode
STEP_LIMIT = 5000


# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)
# def step(position, velocity, action):
#     new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
#     new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
#     new_position = position + new_velocity
#     new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
#     reward = -1.0
#     if new_position == POSITION_MIN:
#         new_velocity = 0.0
#     return new_position, new_velocity, reward


# accumulating trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @return: new trace for convenience
def accumulating_trace(trace, active_tiles, lam, _=None):
    trace *= lam * DISCOUNT
    trace[active_tiles] += 1
    return trace


# replacing trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @return: new trace for convenience
def replacing_trace(trace, activeTiles, lam):
    active = np.in1d(np.arange(len(trace)), activeTiles)
    trace[active] = 1
    trace[~active] *= lam * DISCOUNT
    return trace


# replacing trace update rule, 'clearing' means set all tiles corresponding to non-selected actions to 0
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @clearingTiles: tiles to be cleared
# @return: new trace for convenience
def replacing_trace_with_clearing(trace, active_tiles, lam, clearing_tiles):
    active = np.in1d(np.arange(len(trace)), active_tiles)
    trace[~active] *= lam * DISCOUNT
    trace[clearing_tiles] = 0
    trace[active] = 1
    return trace


# dutch trace update rule
# @trace: old trace (will be modified)
# @activeTiles: current active tile indices
# @lam: lambda
# @alpha: step size for all tiles
# @return: new trace for convenience
def dutch_trace(trace, active_tiles, lam, alpha):
    coef = 1 - alpha * DISCOUNT * lam * np.sum(trace[active_tiles])
    trace *= DISCOUNT * lam
    trace[active_tiles] += coef
    return trace


# wrapper class for Sarsa(lambda)
class Sarsa:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, step_size, lam, trace_update=accumulating_trace, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.trace_update = trace_update
        self.lam = lam

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # trace for each tile
        self.trace = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def get_active_tiles(self, mountain_car, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However, positionScale * position_min is a constant, so it's ok to ignore it.
        state = mountain_car.getState()
        return tiles(self.hash_table, self.num_of_tilings,
                             [self.position_scale * state[0], self.velocity_scale * state[1]], [action])

    # estimate the value of given state and action
    def value(self, mountain_car, action):
        if mountain_car.isTerminal():
            return 0.0
        return np.sum(self.weights[self.get_active_tiles(mountain_car, action)])

    # learn with given state, action and target
    def learn(self, state, action, target):
        active_tiles = self.get_active_tiles(state, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = target - estimation
        if self.trace_update == accumulating_trace or self.trace_update == replacing_trace:
            self.trace_update(self.trace, active_tiles, self.lam)
        elif self.trace_update == dutch_trace:
            self.trace_update(self.trace, active_tiles, self.lam, self.step_size)
        elif self.trace_update == replacing_trace_with_clearing:
            clearing_tiles = []
            for act in ACTIONS:
                if act != action:
                    clearing_tiles.extend(self.get_active_tiles(state, act))
            self.trace_update(self.trace, active_tiles, self.lam, clearing_tiles)
        else:
            raise Exception('Unexpected Trace Type')
        self.weights += self.step_size * delta * self.trace


# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(mountain_car, valueFunction):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(mountain_car, action))
    return np.argmax(values) - 1


# play Mountain Car for one episode based on given method @evaluator
# @return: total steps in this episode
def play(mountain_car, evaluator):
    mountain_car.reset()
    state = mountain_car.getState()
    action = get_action(mountain_car, evaluator)
    steps = 1

    while steps < STEP_LIMIT:
        if mountain_car.isTerminal():
            return steps

        next_state = mountain_car.update(action)
        next_action = get_action(mountain_car, evaluator)

        target = -1 + DISCOUNT * evaluator.value(mountain_car, next_action)
        evaluator.learn(state, action, target)

        state = next_state
        action = next_action
        steps += 1

    print('Step Limit Exceeded!')
    return steps


# figure 12.10, effect of the lambda and alpha on early performance of Sarsa(lambda)
def figure_12_10():
    mountain_car = sim.MountainCar()
    runs = 20
    episodes = 1
    alphas = np.arange(1, 8) / 4.0
    lams = [0.99, 0.9, 0.5, 0]

    steps = np.zeros((len(lams), len(alphas), runs, episodes))
    for lamInd, lam in enumerate(lams):
        for alphaInd, alpha in enumerate(alphas):
            for run in tqdm(range(runs)):
                evaluator = Sarsa(alpha, lam, replacing_trace)
                for ep in range(episodes):
                    steps[lamInd, alphaInd, run, ep] = play(mountain_car, evaluator)

    # average over episodes
    steps = np.mean(steps, axis=3)

    # average over runs
    steps = np.mean(steps, axis=2)

    for lamInd, lam in enumerate(lams):
        plt.plot(alphas, steps[lamInd, :], label='lambda = %s' % (str(lam)))
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.ylim([180, 300])
    plt.legend()

    plt.savefig('images/figure_12_10.png')
    plt.close()


# figure 12.11, summary comparision of Sarsa(lambda) algorithms
# I use 8 tilings rather than 10 tilings
def figure_12_11():
    mountain_car = sim.MountainCar()
    traceTypes = [dutch_trace, replacing_trace, replacing_trace_with_clearing, accumulating_trace]
    alphas = np.arange(0.2, 2.2, 0.2)
    episodes = 1
    runs = 20
    lam = 0.9
    rewards = np.zeros((len(traceTypes), len(alphas), runs, episodes))

    for traceInd, trace in enumerate(traceTypes):
        for alphaInd, alpha in enumerate(alphas):
            for run in tqdm(range(runs)):
                evaluator = Sarsa(alpha, lam, trace)
                for ep in range(episodes):
                    if trace == accumulating_trace and alpha > 0.6:
                        steps = STEP_LIMIT
                    else:
                        steps = play(mountain_car, evaluator)
                    rewards[traceInd, alphaInd, run, ep] = -steps

    # average over episodes
    rewards = np.mean(rewards, axis=3)

    # average over runs
    rewards = np.mean(rewards, axis=2)

    for traceInd, trace in enumerate(traceTypes):
        plt.plot(alphas, rewards[traceInd, :], label=trace.__name__)
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged rewards pre episode')
    plt.ylim([-550, -150])
    plt.legend()

    plt.savefig('images/figure_12_11.png')
    plt.close()


if __name__ == '__main__':
    figure_12_10()
    figure_12_11()
