import random
import numpy as np

n = 0
bins, t_bins = [], []


def init(m):
    global n, bins, t_bins
    n = m
    bins = [
        [0, *np.linspace(np.pi / 12, 2 * np.pi - np.pi / 12, n - 2), 2 * np.pi],  # theta bins
        [*np.linspace(-4, -1, n // 2), 0, *np.linspace(1, 4, n // 2)],  # omega bins
        np.linspace(-4.5, 4.5, n),  # x bins
        [*np.linspace(-4, -1, n // 2), *np.linspace(1, 4, n // 2)]  # velocity bins
    ]
    t_bins = [
        np.linspace(-np.pi / 12, np.pi / 12, n),  # theta bins
        np.linspace(-1, 1, n),  # omega bins
        np.linspace(-4.5, 4.5, 3),  # x bins
        np.linspace(-1, 1, n)  # velocity bins
    ]


def discretize(state):
    state[0] = state[0] % (2 * np.pi)
    if state[0] < bins[0][1]:
        state[0] = 0
    elif state[0] > bins[0][n - 2]:
        state[0] = n - 1
    else:
        delta = (bins[0][2] - bins[0][1]) / 2
        for i in range(1, n - 1):
            if state[0] < bins[0][i] + delta:
                state[0] = i
                break

    delta = (bins[1][1] - bins[1][0]) / 2
    for i in range(n // 2 - 1):
        if state[1] < bins[1][i] + delta:
            state[1] = i
            break
    else:
        if state[1] < bins[1][n // 2 - 1]:
            state[1] = n // 2 - 1
        elif state[1] < bins[1][n // 2 + 1]:
            state[1] = n // 2
        else:
            for i in range(n // 2 + 1, n):
                if state[1] < bins[1][i] + delta:
                    state[1] = i
                    break
            else:
                state[1] = n - 1

    delta = (bins[2][1] - bins[2][0]) / 2
    for i in range(n):
        if state[2] < bins[2][i] + delta:
            state[2] = i
            break

    delta = (bins[3][1] - bins[3][0]) / 2
    for i in range(n // 2 - 1):
        if state[3] < bins[3][i] + delta:
            state[3] = i
            break
    else:
        if state[3] < 0:
            state[3] = n // 2 - 1
        elif state[3] < bins[3][n // 2]:
            state[3] = n // 2
        else:
            for i in range(n // 2, n - 1):
                if state[3] < bins[3][i] + delta:
                    state[3] = i
                    break
            else:
                state[3] = n - 2

    return state[0], state[1], state[2], state[3]


def rand(state):
    delta = (bins[0][2] - bins[0][1]) / 2
    if state[0] == 0:
        state[0] = random.uniform(bins[0][0], bins[0][1])
    elif state[0] == 1:
        state[0] = random.uniform(bins[0][1], bins[0][1] + delta)
    elif state[0] == n - 2:
        state[0] = random.uniform(bins[0][n - 2] - delta, bins[0][n - 2])
    elif state[0] == n - 1:
        state[0] = random.uniform(bins[0][n - 2], bins[0][n - 1])
    else:
        state[0] = random.uniform(bins[0][state[0]] - delta, bins[0][state[0]] + delta)
    state[0] = (state[0] + np.pi) % (2 * np.pi) - np.pi

    delta = (bins[1][1] - bins[1][0]) / 2
    if state[1] == n // 2:
        state[1] = random.uniform(bins[1][n // 2 - 1], bins[1][n // 2 + 1])
    elif state[1] == n // 2 - 1:
        state[1] = random.uniform(bins[1][n // 2 - 1] - delta, bins[1][n // 2 - 1])
    elif state[1] == n // 2 + 1:
        state[1] = random.uniform(bins[1][n // 2 + 1], bins[1][n // 2 + 1] + delta)
    else:
        state[1] = random.uniform(bins[1][state[1]] - delta, bins[1][state[1]] + delta)

    delta = bins[2][2] / 2
    if state[2] == 0:
        state[2] = random.uniform(bins[2][0], bins[2][0] + delta)
    elif state[2] == n - 1:
        state[2] = random.uniform(bins[2][n - 1] - delta, bins[2][n - 1])
    else:
        state[2] = random.uniform(bins[2][state[2]] - delta, bins[2][state[2]] + delta)

    delta = (bins[3][1] - bins[3][0]) / 2
    delta2 = (bins[3][n // 2] - bins[3][n // 2 - 1]) / 2
    if state[3] == n // 2 - 1:
        state[3] = random.uniform(bins[3][n // 2 - 1] - delta, bins[3][n // 2 - 1] + delta2)
    elif state[3] == n // 2:
        state[3] = random.uniform(bins[3][n // 2] - delta2, bins[3][n // 2] + delta)
    else:
        state[3] = random.uniform(bins[3][state[3]] - delta, bins[3][state[3]] + delta)

    return state[0], state[1], state[2], state[3]


def t_discretize(state):
    for j in [0, 1, 3]:
        if state[j] > t_bins[j][n - 1]:
            state[j] = n - 1
        else:
            delta = (t_bins[j][1] - t_bins[j][0]) / 2
            for i in range(n):
                if state[j] < t_bins[j][i] + delta:
                    state[j] = i
                    break

    delta = (t_bins[2][1] - t_bins[2][0]) / 2
    if state[2] < t_bins[2][0] + delta:
        state[2] = 0
    elif state[2] < t_bins[2][1] + delta:
        state[2] = 1
    else:
        state[2] = 2

    return state[0], state[1], state[2], state[3]


def t_rand(state):
    for i in [0, 1, 3]:
        delta = t_bins[i][2] / 2
        if state[i] == 0:
            state[i] = random.uniform(t_bins[i][0], t_bins[i][0] + delta)
        elif state[i] == n - 1:
            state[i] = random.uniform(t_bins[i][n - 1] - delta, t_bins[i][n - 1])
        else:
            state[i] = random.uniform(t_bins[i][state[i]] - delta, t_bins[i][state[i]] + delta)

    delta = t_bins[2][2] / 2
    if state[2] == 0:
        state[2] = random.uniform(t_bins[2][0], t_bins[2][0] + delta)
    elif state[2] == 2:
        state[2] = random.uniform(t_bins[2][2] - delta, t_bins[2][2])
    else:
        state[2] = random.uniform(t_bins[2][1] - delta, t_bins[2][1] + delta)

    return state[0], state[1], state[2], state[3]
