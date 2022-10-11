import random
import struct

import numpy as np

n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE, bins = 0, [], 0, 0, 0, 0, 0, []


def init(a, b, c, d, e, f, g):
    global n
    global u
    global epsilon
    global alpha
    global gamma
    global SET_STATE
    global APPLY_FORCE
    global bins

    n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE = a, b, c, d, e, f, g
    bins = [
        [0, *np.linspace(np.pi / 18, 2 * np.pi - np.pi / 18, n - 2), 2 * np.pi],  # theta bins
        [*np.linspace(-10, -1, n // 2), 0, *np.linspace(1, 10, n // 2)],  # omega bins
        np.linspace(-4.5, 4.5, n),  # x bins
        [*np.linspace(-10, -1, n // 2), *np.linspace(1, 10, n // 2)]  # velocity bins
    ]


def sarsa(Q, episodes, socket, steps=150):
    episodes2 = []
    for (t0, w0, x0, v0) in episodes:
        state = rand([t0, w0, x0, v0])
        socket.send(struct.pack('iffff', SET_STATE, *state))
        _ = socket.recv()

        A = e_greedy(Q, t0, w0, x0, v0)
        t, w, x, v = t0, w0, x0, v0
        for _ in range(0, steps):
            t, w, x, v, Q, A = updateQ(socket, Q, t, w, x, v, A)
            if (t == 0 or t == n - 1) and w == n // 2:
                break
        else:
            episodes2.append([t0, w0, x0, v0])

    return Q, episodes2


def updateQ(socket, Q, t, w, x, v, A):
    socket.send(struct.pack('if', APPLY_FORCE, u[A]))
    tp, wp, xp, vp = struct.unpack('ffff', socket.recv())

    # R = (-1 + abs(wp) / 4) * abs(tp) / np.pi - abs(wp) / 8
    tp, wp, xp, vp = discretize([tp, wp, xp, vp])
    R = -1
    if (t == 0 or t == n - 1) and w == n // 2:
        R = 0

    Ap = e_greedy(Q, tp, wp, xp, vp)

    Q[t][w][x][v][A] += alpha * (R + gamma * Q[tp][wp][xp][vp][Ap] - Q[t][w][x][v][A])
    return tp, wp, xp, vp, Q, Ap


def e_greedy(Q, t, w, x, v):
    if random.random() <= epsilon:
        return random.randint(0, 1)

    return int(Q[t][w][x][v][0] < Q[t][w][x][v][1])


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
            for i in range(n // 2, n):
                if state[3] < bins[3][i] + delta:
                    state[3] = i
                    break

    return state[0], state[1], state[2], state[3]


def rand(state):
    delta = (bins[0][2] - bins[0][1]) / 2
    if state[0] == 0:
        state[0] = random.uniform(bins[0][0], bins[0][1])
    elif state[0] == 1:
        state[0] = random.uniform(bins[0][1], bins[0][1] + delta)
    if state[0] == n - 2:
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
