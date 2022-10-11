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
        np.linspace(-np.pi / 18, np.pi / 18, n),  # theta bins
        np.linspace(-1, 1, n),  # omega bins
        np.linspace(-4.5, 4.5, 3),  # x bins
        np.linspace(-1, 1, n)  # velocity bins
    ]


def ql(Q, episodes, socket, steps=150):
    episodes2 = []
    for (t0, w0, x0, v0) in episodes:
        state = rand([t0, w0, x0, v0])
        socket.send(struct.pack('iffff', SET_STATE, *state))
        _ = socket.recv()

        t, w, x, v = t0, w0, x0, v0
        for _ in range(0, steps):
            A = e_greedy(Q, t, w, x, v)
            t, w, x, v, Q, _ = updateQ(socket, Q, t, w, x, v, A)
            if t == 0 or t == n - 1:
                episodes2.append([t0, w0, x0, v0])
                break

    return Q, episodes2


def updateQ(socket, Q, t, w, x, v, A):
    socket.send(struct.pack('if', APPLY_FORCE, u[A]))
    tp, wp, xp, vp = struct.unpack('ffff', socket.recv())

    # R = (-1 + abs(wp) / 4) * abs(tp) / np.pi - abs(wp) / 8
    tp, wp, xp, vp = discretize([tp, wp, xp, vp])
    R = -1
    if (t == 0 or t == n - 1) and w == n // 2:
        R = 0

    Ap = e_greedy(Q, tp, wp, xp, vp, 0)

    Q[t][w][x][v][A] += alpha * (R + gamma * Q[tp][wp][xp][vp][Ap] - Q[t][w][x][v][A])
    return tp, wp, xp, vp, Q, Ap


def e_greedy(Q, t, w, x, v, e=epsilon):
    if random.random() <= e:
        return random.randint(0, 1)

    return int(Q[t][w][x][v][0] < Q[t][w][x][v][1])


def discretize(state):
    for j in [0, 1, 3]:
        if state[j] > bins[j][n - 1]:
            state[j] = n - 1
        else:
            delta = (bins[j][1] - bins[j][0]) / 2
            for i in range(n):
                if state[j] < bins[j][i] + delta:
                    state[j] = i
                    break

    delta = (bins[2][1] - bins[2][0]) / 2
    if state[2] < bins[2][0] + delta:
        state[2] = 0
    elif state[2] < bins[2][1] + delta:
        state[2] = 1
    else:
        state[2] = 2

    return state[0], state[1], state[2], state[3]


def rand(state):
    for i in [0, 1, 3]:
        delta = bins[i][2] / 2
        if state[i] == 0:
            state[i] = random.uniform(bins[i][0], bins[i][0] + delta)
        elif state[i] == n - 1:
            state[i] = random.uniform(bins[i][n - 1] - delta, bins[i][n - 1])
        else:
            state[i] = random.uniform(bins[i][state[i]] - delta, bins[i][state[i]] + delta)

    delta = bins[2][2] / 2
    if state[2] == 0:
        state[2] = random.uniform(bins[2][0], bins[2][0] + delta)
    elif state[2] == 2:
        state[2] = random.uniform(bins[2][2] - delta, bins[2][2])
    else:
        state[2] = random.uniform(bins[2][1] - delta, bins[2][1] + delta)

    return state[0], state[1], state[2], state[3]
