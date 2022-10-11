import struct
from os import remove
from threading import Thread

import numpy as np
import zmq

import ql
import sarsa
from server import main as run

n = 5  # Number of bins (odd). Try to keep low: Q = O(n^4)
u = [-10, 10]  # force bins
epsilon = 0.1  # for e-greedy algorithm
alpha = 0.02
gamma = 1  # discount

APPLY_FORCE = 0
SET_STATE = 1
TRAINING = 3


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', TRAINING))
    _ = socket.recv()

    q = [[[[[-1 for _ in range(2)] for _ in range(n - 1)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    Q = [[[[[-1 for _ in range(2)] for _ in range(n - 1)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    Q_new = [[[[0 for _ in range(n - 1)] for _ in range(n)] for _ in range(n)] for _ in range(n)]

    remove('Q.npy')

    q = do_sarsa(q, 100, socket)
    Q = do_ql(Q, 100, socket)
    with open('Q.npy', 'wb') as f:
        for (t, w, x, v), _ in np.ndenumerate(np.array(Q_new)):
            Q_new[t][w][x][v] = int((Q[t][w][x][v][0] + q[t][w][x][v][0]) < (Q[t][w][x][v][1] + q[t][w][x][v][1]))
        np.save(f, np.array(Q_new))


def do_sarsa(Q, steps, socket):
    sarsa.init(n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE)

    for x in range(n):
        for v in range(n - 1):
            Q[0][n // 2][x][v][0] = 0
            Q[0][n // 2][x][v][1] = 0
            Q[n - 1][n // 2][x][v][0] = 0
            Q[n - 1][n // 2][x][v][1] = 0

    i = 0
    count = 0
    episodes = []
    for (t, w, x, v, _), _ in np.ndenumerate(np.array(Q)):
        if i % 2 == 0 and not (t == 1 and w == n // 2):
            episodes.append([t, w, x, v])
            count += 1
        i += 1

    i = 1
    last = 0
    print("SARSA Episodes:", count)
    while len(episodes) > count * 0.05:  # at least 95% of episodes should reach terminal state
        Q_sarsa, episodes = sarsa.sarsa(Q, episodes, socket, steps=steps)
        if len(episodes) != last:
            print("Iteration:", i, "|", "Failed Episodes:", len(episodes))
            last = len(episodes)
        i += 1

    return Q


def do_ql(Q, steps, socket):
    ql.init(n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE)

    for x in range(n):
        for v in range(n - 1):
            Q[0][n // 2][x][v][0] = 0
            Q[0][n // 2][x][v][1] = 0
            Q[n - 1][n // 2][x][v][0] = 0
            Q[n - 1][n // 2][x][v][1] = 0

    i = 0
    count = 0
    episodes = []
    for (t, w, x, v, _), _ in np.ndenumerate(np.array(Q)):
        if i % 2 == 0:
            episodes.append([t, w, x, v])
            count += 1
        i += 1

    i = 1
    last = 0
    print("QL Episodes:", count)
    while len(episodes) > count * 0.05:  # at least 95% of episodes should stay in terminal state
        Q, episodes = ql.ql(Q, episodes, socket, steps=steps)
        if len(episodes) != last:
            print("Iteration:", i, "|", "Failed Episodes:", len(episodes))
            last = len(episodes)
        i += 1

    return Q


if __name__ == "__main__":
    th1 = Thread(target=main)
    th2 = Thread(target=run, daemon=True)

    th1.start()
    th2.start()
