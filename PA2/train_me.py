import random
import struct
from os import remove
from threading import Thread

import numpy as np
import zmq

from server import main as run

n = 7  # Number of bins (odd). Try to keep low: Q = O(n^4)
sarsa_bins = [
    [0, *np.linspace(np.pi / 18, 2 * np.pi - np.pi / 18, n - 2), 2 * np.pi],  # theta bins
    [*np.linspace(-10, -1, n // 2), 0, *np.linspace(1, 10, n // 2)],  # omega bins
    np.linspace(-4.5, 4.5, n),  # x bins
    [*np.linspace(-10, -1, n // 2), *np.linspace(1, 10, n // 2)]  # velocity bins
]
ql_bins = [
    np.linspace(-np.pi / 18, np.pi / 18, n),  # theta bins
    np.linspace(-1, 1, n),  # omega bins
    np.linspace(-4.5, 4.5, 3),  # x bins
    np.linspace(-1, 1, n)  # velocity bins
]
u = [-10, 10]  # force bins

epsilon = 0.05  # for e-greedy algorithm
alpha = 0.02
gamma = 1  # discount

APPLY_FORCE = 0
SET_STATE = 1
TRAINING = 3

random.seed(0)


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', TRAINING))
    _ = socket.recv()

    remove('Q_sarsa.npy')
    remove('Q_ql.npy')

    Q_sarsa = [[[[[-1 for _ in range(2)] for _ in range(n - 1)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for x in range(n):
        for v in range(n - 1):
            Q_sarsa[0][n // 2][x][v][0] = 0
            Q_sarsa[0][n // 2][x][v][1] = 0
            Q_sarsa[n - 1][n // 2][x][v][0] = 0
            Q_sarsa[n - 1][n // 2][x][v][1] = 0

    i = 0
    count = 0
    sarsa_episodes = []
    for (t, w, x, v, _), _ in np.ndenumerate(np.array(Q_sarsa)):
        if i % 2 == 0 and not (t == 1 and w == n // 2):
            sarsa_episodes.append([t, w, x, v])
            count += 1
        i += 1

    i = 1
    print("SARSA Episodes:", count)
    while len(sarsa_episodes) > count * 0.05:  # at least 95% of episodes should reach terminal state
        Q_sarsa, sarsa_episodes = sarsa(Q_sarsa, sarsa_episodes, socket, steps=150)
        print("Iteration:", i, "|", "Failed Episodes:", len(sarsa_episodes))
        i += 1

    Q_new = [[[[0 for _ in range(n - 1)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    with open('Q_sarsa.npy', 'wb') as f:
        for (t, w, x, v), _ in np.ndenumerate(np.array(Q_new)):
            Q_new[t][w][x][v] = int(Q_sarsa[t][w][x][v][0] < Q_sarsa[t][w][x][v][1])
        np.save(f, np.array(Q_new))

    Q_ql = [[[[[0 for _ in range(2)] for _ in range(n)] for _ in range(3)] for _ in range(n)] for _ in range(n)]

    i = 0
    count = 0
    ql_episodes = []
    for (t, w, x, v, _), _ in np.ndenumerate(np.array(Q_ql)):
        if i % 2 == 0 and not (t == 1 and w == n // 2):
            ql_episodes.append([t, w, x, v])
            count += 1
        i += 1

    i = 1
    print("\nQL Episodes:", count)
    while len(ql_episodes) > count * 0.05:  # it can always reach 100%, but 95% takes much less time
        Q_ql, ql_episodes = ql(Q_ql, ql_episodes, socket, steps=150)
        print("Iteration:", i, "|", "Failed Episodes:", len(ql_episodes))
        i += 1

    Q_new = [[[[0 for _ in range(n)] for _ in range(3)] for _ in range(n)] for _ in range(n)]
    with open('Q_ql.npy', 'wb') as f:
        for (t, w, x, v), _ in np.ndenumerate(np.array(Q_new)):
            Q_new[t][w][x][v] = int(Q_ql[t][w][x][v][0] < Q_ql[t][w][x][v][1])
        np.save(f, np.array(Q_new))


def sarsa(Q, episodes, socket, steps=150):
    episodes2 = []
    for (t0, w0, x0, v0) in episodes:
        state = sarsa_rand([t0, w0, x0, v0])
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


def ql(Q, episodes, socket, steps=150):
    episodes2 = []
    for (t0, w0, x0, v0) in episodes:
        state = sarsa_rand([t0, w0, x0, v0])
        socket.send(struct.pack('iffff', SET_STATE, *state))
        _ = socket.recv()

        t, w, x, v = t0, w0, x0, v0
        for _ in range(0, steps):
            A = e_greedy(Q, t, w, x, v)
            t, w, x, v, Q, _ = updateQ(socket, Q, t, w, x, v, A, 0)  # e = 0 means always choose max A
            if t == 0 or t == n - 1:
                episodes2.append([t0, w0, x0, v0])
                break

    return Q, episodes2


def updateQ(socket, Q, t, w, x, v, A, e=epsilon):
    socket.send(struct.pack('if', APPLY_FORCE, u[A]))
    tp, wp, xp, vp = struct.unpack('ffff', socket.recv())

    # R = (-1 + abs(wp) / 4) * abs(tp) / np.pi - abs(wp) / 8
    tp, wp, xp, vp = sarsa_discretize([tp, wp, xp, vp])
    R = -1
    if (t == 0 or t == n - 1) and w == n // 2:
        R = 0

    Ap = e_greedy(Q, tp, wp, xp, vp, e)

    Q[t][w][x][v][A] += alpha * (R + gamma * Q[tp][wp][xp][vp][Ap] - Q[t][w][x][v][A])
    return tp, wp, xp, vp, Q, Ap


def e_greedy(Q, t, w, x, v, e=epsilon):
    if random.random() <= e:
        return random.randint(0, 1)

    return int(Q[t][w][x][v][0] < Q[t][w][x][v][1])


def sarsa_discretize(state):
    state[0] = state[0] % (2 * np.pi)
    if state[0] < sarsa_bins[0][1]:
        state[0] = 0
    elif state[0] > sarsa_bins[0][n - 2]:
        state[0] = n - 1
    else:
        delta = (sarsa_bins[0][2] - sarsa_bins[0][1]) / 2
        for i in range(1, n - 1):
            if state[0] < sarsa_bins[0][i] + delta:
                state[0] = i
                break

    delta = (sarsa_bins[1][1] - sarsa_bins[1][0]) / 2
    for i in range(n // 2 - 1):
        if state[1] < sarsa_bins[1][i] + delta:
            state[1] = i
            break
    else:
        if state[1] < sarsa_bins[1][n // 2 - 1]:
            state[1] = n // 2 - 1
        elif state[1] < sarsa_bins[1][n // 2 + 1]:
            state[1] = n // 2
        else:
            for i in range(n // 2 + 1, n):
                if state[1] < sarsa_bins[1][i] + delta:
                    state[1] = i
                    break

    delta = (sarsa_bins[2][1] - sarsa_bins[2][0]) / 2
    for i in range(n):
        if state[2] < sarsa_bins[2][i] + delta:
            state[2] = i
            break

    delta = (sarsa_bins[3][1] - sarsa_bins[3][0]) / 2
    for i in range(n // 2 - 1):
        if state[3] < sarsa_bins[3][i] + delta:
            state[3] = i
            break
    else:
        if state[3] < 0:
            state[3] = n // 2 - 1
        elif state[3] < sarsa_bins[3][n // 2]:
            state[3] = n // 2
        else:
            for i in range(n // 2, n):
                if state[3] < sarsa_bins[3][i] + delta:
                    state[3] = i
                    break

    return state[0], state[1], state[2], state[3]


def ql_discretize(state):
    for j in [0, 1, 3]:
        if state[j] > sarsa_bins[j][n - 1]:
            state[j] = n - 1
        else:
            delta = (sarsa_bins[j][1] - sarsa_bins[j][0]) / 2
            for i in range(n):
                if state[j] < sarsa_bins[j][i] + delta:
                    state[j] = i
                    break

    delta = (sarsa_bins[2][1] - sarsa_bins[2][0]) / 2
    if state[2] < sarsa_bins[2][0] + delta:
        state[2] = 0
    elif state[2] < sarsa_bins[2][1] + delta:
        state[2] = 1
    else:
        state[2] = 2

    return state[0], state[1], state[2], state[3]


def sarsa_rand(state):
    if state[0] == 0:
        state[0] = random.uniform(-np.pi, sarsa_bins[0][0])
    elif state[0] == 1:
        state[0] = random.uniform(sarsa_bins[0][0], sarsa_bins[0][1])
    else:
        state[0] = random.uniform(sarsa_bins[0][1], np.pi)

    delta = sarsa_bins[2][2] / 2
    if state[2] == 0:
        state[2] = random.uniform(sarsa_bins[2][0], sarsa_bins[2][0] + delta)
    elif state[2] == 1:
        state[2] = random.uniform(sarsa_bins[2][0] + delta, sarsa_bins[2][1] + delta)
    else:
        state[2] = random.uniform(sarsa_bins[2][1] + delta, sarsa_bins[2][2])

    for i in [1, 3]:
        attr = sarsa_bins[i]
        delta = (attr[n // 2] - attr[n // 2 - 1]) / 2
        state[i] = random.uniform(attr[state[i]] - delta, attr[state[i]] + delta)

    return state[0], state[1], state[2], state[3]


if __name__ == "__main__":
    th1 = Thread(target=main)
    th2 = Thread(target=run, daemon=True)

    th1.start()
    th2.start()
