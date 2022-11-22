#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from MountainCar import MountainCar
# from SarsaLambda import SarsaLambda as sl
from temp import SarsaLambda as sl

DISCOUNT = 1.0
EPSILON = 0
MAX_STEPS = 1000


# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(mountain_car, evaluator):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(mountain_car.actions)
    values = []
    for action in mountain_car.actions:
        values.append(evaluator.value(action))
    return np.argmax(values) - 1


# play Mountain Car for one episode based on given method @evaluator
# @return: total steps in this episode
def play(mountain_car, evaluator):
    mountain_car.reset()
    state = mountain_car.getState()
    action = get_action(mountain_car, evaluator)
    Q = evaluator.value(action)
    steps = 1

    while steps < MAX_STEPS:
        if mountain_car.isTerminal():
            return steps

        next_state = mountain_car.update(action)
        next_action = get_action(mountain_car, evaluator)

        Qp = evaluator.value(next_action)
        delta = -1 + DISCOUNT * Qp - Q
        evaluator.learn(state, action, delta)

        Q = Qp
        state = next_state
        action = next_action
        steps += 1

    return steps


# print learned cost to go
def print_cost(model, value_function, episode, ax):
    grid_size = 40
    positions = np.linspace(model.min_maxes[0], model.min_maxes[1], grid_size)
    velocities = np.linspace(model.min_maxes[2], model.min_maxes[3], grid_size)
    axis_x = []
    axis_y = []
    axis_z = []
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            model.set([position, velocity])
            axis_z.append(value_function.cost_to_go())

    ax.scatter(axis_x, axis_y, axis_z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % episode)


# Figure 1, steps per episode
def figure_1():
    runs = 20
    episodes = 50
    orders = [3]  # , 5, 7]

    mountain_car = MountainCar()

    steps = np.zeros((len(orders), episodes))
    with tqdm(total=runs * episodes * len(orders), ncols=100) as progress:
        for i in range(len(orders)):
            for run in range(runs):
                sarsa_lam = sl(0.9, 0.4, mountain_car, orders[i])
                for episode in range(episodes):
                    steps[i, episode] += play(mountain_car, sarsa_lam)
                    progress.update()

    # average over episodes
    steps /= runs

    for i, order in enumerate(orders):
        plt.plot(steps[i, :], label='Order = %d' % order)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()

    plt.savefig('images/figure_1.png')
    plt.close()


# Figure 2, cost to go in a single run
def figure_2(order):
    episodes = 100
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(projection='3d')

    mountain_car = MountainCar()
    sarsa_lam = sl(0.9, 0.4, mountain_car, order)

    for _ in tqdm(range(episodes), ncols=100):
        play(mountain_car, sarsa_lam)

    print_cost(mountain_car, sarsa_lam, episodes, axes)

    plt.savefig('images/figure_2_O(%d).png' % order)
    plt.close()


if __name__ == '__main__':
    # figure_1()
    figure_2(3)
    # figure_2(5)
    # figure_2(7)
