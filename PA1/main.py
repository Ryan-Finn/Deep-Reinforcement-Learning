import matplotlib.pyplot as plt
import numpy as np
from draw import draw_V, draw_Pi

GRID = {
    "world_size": (4, 4),
    "start": [0, 0],
    "goal": [3, 3],
    "tele": [2, 0],
}
GAMMA = 0.95
THETA = 1e-4

ACTIONS = [np.array([0, -1]),  # left
           np.array([-1, 0]),  # up
           np.array([0, 1]),  # right
           np.array([1, 0])]  # down


# Make new policy
def new_policy(init=0.0):
    pi = np.empty(GRID["world_size"], dtype=object)

    for (i, j), _ in np.ndenumerate(pi):
        pi[i, j] = init * np.ones(GRID["world_size"][0])

    return pi


# Get next state and reward from current state and action
def step(state, action):
    next_state = (np.array(state) + action).tolist()
    i, j = next_state

    if i < 0 or i >= GRID["world_size"][0] or j < 0 or j >= GRID["world_size"][1]:
        if state == GRID["goal"]:
            return GRID["goal"], 0
        if state == GRID["tele"]:
            return GRID["goal"], -2.0
        return state, -1.0

    return next_state, -1.0


def evaluate(pi):
    v = np.zeros(GRID["world_size"])

    while True:
        v_prime = np.zeros(GRID["world_size"])

        for (i, j), _ in np.ndenumerate(pi):
            for k, action in enumerate(ACTIONS):
                (i_prime, j_prime), reward = step([i, j], action)
                v_prime[i, j] += pi[i, j][k] * (reward + GAMMA * v[i_prime, j_prime])

        if abs(v - v_prime).max() < THETA:
            return v_prime

        v = v_prime


def improve(v):
    pi = new_policy()
    pi_d = new_policy()

    for (i, j), _ in np.ndenumerate(pi):
        best_a = -np.inf * np.ones(GRID["world_size"][0])
        for k, action in enumerate(ACTIONS):
            (i_prime, j_prime), _ = step([i, j], action)
            best_a[k] = v[i_prime, j_prime]

        indices = np.where(best_a == max(best_a))[0]
        for ind in indices:
            pi[i, j][ind] = 1 / int(len(indices))
        pi_d[i, j][indices[0]] = 1

    return pi, pi_d


def main():
    v = evaluate(new_policy(0.25))

    while True:
        pi_star, pi_d = improve(v)
        v_star = evaluate(pi_star)

        if abs(v - v_star).max() < THETA:
            break
        v = v_star

    print("A Deterministic Policy:")
    print(" L   U   R   D")
    for _, k in np.ndenumerate(pi_d):
        print(*k)

    print("\nA Stochastic Policy:")
    print(" L   U   R   D")
    number = 1
    for _, k in np.ndenumerate(pi_star):
        print(*k)
        number *= int(len(np.where(k != 0)[0]))

    print("\nNumber of Deterministic Policies:", number)

    draw_V(np.round(v_star, decimals=2), GRID)
    plt.savefig('../PA1/images/v-star.png')
    plt.close()
    draw_Pi(pi_star, GRID)
    plt.savefig('../PA1/images/pi-star.png')
    plt.close()


if __name__ == '__main__':
    main()
