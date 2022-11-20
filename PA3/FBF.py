#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')

# # of states except for terminal states
N_STATES = 1000

# all states
STATES = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100


def compute_true_value():
    # true state value, just a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            for action in ACTIONS:
                for sp in range(1, STEP_RANGE + 1):
                    sp *= action
                    next_state = state + sp
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    return true_value


# take an @action at @state, return new state and reward for this transition
def step(state, action):
    sp = np.random.randint(1, STEP_RANGE + 1)
    sp *= action
    state += sp
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward


# get an action, following random policy
def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1


# a wrapper class for polynomial / Fourier -based value function
POLYNOMIAL_BASES = 0
FOURIER_BASES = 1


class BasesValueFunction:
    # @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order):
        self.order = order
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        for n in range(order + 1):
            self.bases.append(lambda s, i=n: np.cos(i * np.pi * s))

    # get the value of @state
    def value(self, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get derivative value
        derivative_value = np.asarray([func(state) for func in self.bases])
        self.weights += delta * derivative_value


# gradient Monte Carlo algorithm
# @value_function: an instance of class ValueFunction
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradient_monte_carlo(value_function, alpha, distribution=None):
    state = START_STATE
    trajectory = [state]

    # We assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while state not in END_STATES:
        action = get_action()
        next_state, reward = step(state, action)
        trajectory.append(next_state)
        state = next_state

    # Gradient update for each state in this trajectory
    for state in trajectory[:-1]:
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1


# Figure 9.5, Fourier basis and polynomials
def figure_9_5(true_value):
    runs = 1
    episodes = 1000
    orders = [3, 5, 7]
    alpha = 5e-5

    # track errors for each episode
    errors = np.zeros((len(orders), episodes))
    with tqdm(total=runs * episodes * len(orders), ncols=100) as progress:
        for run in range(runs):
            for i in range(len(orders)):
                value_function = BasesValueFunction(orders[i])
                for episode in range(episodes):
                    # gradient Monte Carlo algorithm
                    gradient_monte_carlo(value_function, alpha)

                    # get state values under current value function
                    state_values = [value_function.value(state) for state in STATES]

                    # get the root-mean-squared error
                    errors[i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))
                    progress.update()

    # average over independent runs
    errors /= runs

    for j in range(len(orders)):
        plt.plot(errors[j, :], label='Order = %d' % orders[j])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('images/figure_9_5.png')
    plt.close()


if __name__ == '__main__':
    figure_9_5(compute_true_value())
