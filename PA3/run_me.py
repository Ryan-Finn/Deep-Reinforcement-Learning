#######################################################################
# Copyright (C)                                                       #
# 2017-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from MountainCar import MountainCar
from SarsaLambda import SarsaLambda as sl

matplotlib.use('Agg')

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
MIN_MAXES = [POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX]

# discount is always 1.0 in these experiments
DISCOUNT = 1.0

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# maximum steps per episode
STEP_LIMIT = 1000


# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(mountain_car, evaluator):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(evaluator.value(mountain_car, action))
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

    # print('Step Limit Exceeded!')
    return steps


# figure 12.10, effect of the lambda and alpha on early performance of Sarsa(lambda)
def figure_12_10():
    mountain_car = MountainCar()
    sarsa_lam = sl(ACTIONS, MIN_MAXES, DISCOUNT)

    runs = 5
    episodes = 50
    alphas = np.arange(1, 8) / 4.0
    lams = [0.99, 0.9, 0.5, 0]

    steps = np.zeros((len(lams), len(alphas), runs, episodes))
    with tqdm(total=runs * episodes * len(lams) * len(alphas), ncols=100) as progress:
        for lamInd, lam in enumerate(lams):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
                    evaluator = sarsa_lam.setEvaluator(alpha, lam)
                    for ep in range(episodes):
                        steps[lamInd, alphaInd, run, ep] = play(mountain_car, evaluator)
                        progress.update()

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


# figure 12.11, summary comparison of Sarsa(lambda) algorithms
# I use 8 tilings rather than 10 tilings
def figure_12_11():
    mountain_car = MountainCar()
    sarsa_lam = sl(ACTIONS, MIN_MAXES, DISCOUNT)

    runs = 5
    episodes = 50
    alphas = np.arange(1, 8) / 4.0
    lam = 0.9

    traceTypes = sarsa_lam.traceTypes
    rewards = np.zeros((len(traceTypes), len(alphas), runs, episodes))

    with tqdm(total=runs * episodes * len(traceTypes) * len(alphas), ncols=100) as progress:
        for traceInd, trace in enumerate(traceTypes):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
                    for ep in range(episodes):
                        evaluator = sarsa_lam.setEvaluator(alpha, lam, trace)
                        if trace == sarsa_lam.accumulating_trace and alpha > 0.6:
                            steps = STEP_LIMIT
                        else:
                            steps = play(mountain_car, evaluator)
                        rewards[traceInd, alphaInd, run, ep] = -steps
                        progress.update()

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
    # figure_12_11()
