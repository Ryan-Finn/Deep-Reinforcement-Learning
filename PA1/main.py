import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 4
START = [0, 0]
GOAL = [3, 3]
TELE = [2, 0]
DISCOUNT = 0.95

# left, up, right, down
ACTIONS_FIGS = ['←', '↑', '→', '↓']
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def new_policy():
    pi = np.empty((WORLD_SIZE, WORLD_SIZE), dtype=object)

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            pi[i, j] = [0.25, 0.25, 0.25, 0.25]

    return pi


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):

        # add state labels
        if [i, j] == GOAL:
            val = str(val) + "\nGOAL"
        if [i, j] == START:
            val = str(val) + "\nSTART"
        if [i, j] == TELE:
            val = str(val) + "\nTELEPORT"

        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

        # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)


def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])

        best_actions = np.where(next_vals == np.max(next_vals))[0]
        val = ''
        for ba in best_actions:
            val += ACTIONS_FIGS[ba]

        # add state labels
        if [i, j] == GOAL:
            val = str(val) + "\nGOAL"
        if [i, j] == START:
            val = str(val) + "\nSTART"
        if [i, j] == TELE:
            val = str(val) + "\nTELEPORT"

        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)


def step(state, action):
    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        if state == GOAL:
            return GOAL, 0
        if state == TELE:
            return GOAL, -2.0
        return state, -1.0

    return next_state, -1.0


def evaluate(pi):
    V = np.zeros((WORLD_SIZE, WORLD_SIZE))

    while True:
        V_prime = np.zeros((WORLD_SIZE, WORLD_SIZE))

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for k, action in enumerate(ACTIONS):
                    (i_prime, j_prime), reward = step([i, j], action)
                    V_prime[i, j] += pi[i, j][k] * (reward + DISCOUNT * V[i_prime, j_prime])
        V_prime[WORLD_SIZE - 1, WORLD_SIZE - 1] = 0

        if abs(V - V_prime).max() < 1e-4:
            return V_prime

        V = V_prime


def improve(V):
    pi = new_policy()

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            best_a = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
            for k, action in enumerate(ACTIONS):
                pi[i, j][k] = 0
                (i_prime, j_prime), _ = step([i, j], action)
                best_a[k] = V[i_prime, j_prime]

            indices = np.where(best_a == max(best_a))[0]
            for ind in indices:
                pi[i, j][ind] = 1 / int(len(indices))

    return pi


def main():
    pi = new_policy()
    V = evaluate(pi)

    while True:
        pi = improve(V)
        V_prime = evaluate(pi)

        if abs(V - V_prime).max() < 1e-4:
            break

        V = V_prime

    number = 1
    for a in pi:
        for b in a:
            print(b)
            number *= int(len(np.where(np.array(b) != 0)[0]))

    print(number)


    draw_image(np.round(V_prime, decimals=2))
    plt.savefig('../PA1/images/v-star.png')
    plt.close()
    draw_policy(V_prime)
    plt.savefig('../PA1/images/pi-star.png')
    plt.close()


if __name__ == '__main__':
    main()
