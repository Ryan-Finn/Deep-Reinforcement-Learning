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

DISCOUNT = 1.0
EPSILON = 0
MAX_STEPS = 1000


# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(mountain_car, evaluator):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(mountain_car.actions)
    values = []
    for action in mountain_car.actions:
        values.append(evaluator.value(mountain_car, action))
    return np.argmax(values) - 1


# play Mountain Car for one episode based on given method @evaluator
# @return: total steps in this episode
def play(mountain_car, evaluator):
    mountain_car.reset()
    state = mountain_car.getState()
    action = get_action(mountain_car, evaluator)
    steps = 1

    while steps < MAX_STEPS:
        if mountain_car.isTerminal():
            return steps

        next_state = mountain_car.update(action)
        next_action = get_action(mountain_car, evaluator)

        target = -1 + DISCOUNT * evaluator.value(mountain_car, next_action)
        evaluator.learn(state, action, target)

        state = next_state
        action = next_action
        steps += 1

    return steps


# figure 12.10, effect of the lambda and alpha on early performance of Sarsa(lambda)
def figure_12_10():
    mountain_car = MountainCar()
    sarsa_lam = sl(mountain_car.actions, mountain_car.min_maxes, DISCOUNT)

    runs = 5
    episodes = 50
    alphas = np.arange(1, 8) / 4.0
    lam = 0.9

    steps = np.zeros((len(alphas), runs, episodes))
    with tqdm(total=runs * episodes * len(alphas), ncols=100) as progress:
        for alphaInd, alpha in enumerate(alphas):
            for run in range(runs):
                evaluator = sarsa_lam.setEvaluator(alpha, lam)
                for ep in range(episodes):
                    steps[alphaInd, run, ep] = play(mountain_car, evaluator)
                    progress.update()

    # average over episodes
    steps = np.mean(steps, axis=2)

    # average over runs
    steps = np.mean(steps, axis=1)

    plt.plot(alphas, steps[:], label='lambda = %s' % (str(lam)))
    plt.xlabel('alpha * # of tilings (8)')
    plt.ylabel('averaged steps per episode')
    plt.legend()

    plt.savefig('images/figure_12_10.png')
    plt.close()


# figure 12.11, summary comparison of Sarsa(lambda) algorithms
# I use 8 tilings rather than 10 tilings
def figure_12_11():
    mountain_car = MountainCar()
    sarsa_lam = sl(mountain_car.actions, mountain_car.min_maxes, DISCOUNT)

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
                            steps = MAX_STEPS
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
    plt.ylabel('averaged rewards per episode')
    plt.legend()

    plt.savefig('images/figure_12_11.png')
    plt.close()


if __name__ == '__main__':
    figure_12_10()
    figure_12_11()
