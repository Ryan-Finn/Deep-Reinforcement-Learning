import struct
# from os import remove
from threading import Thread

import numpy as np
import zmq

import ql
import sarsa
from server import main as run

n = 5  # Number of bins (odd). Try to keep low: Q = O(n^4)
u = [-10, 10]  # force bins
epsilon = 0.05  # for e-greedy algorithm
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

    sarsa.init(n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE)
    ql.init(n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE)

    # remove('Q_sarsa.npy')
    # remove('Q_ql.npy')

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
        Q_sarsa, sarsa_episodes = sarsa.sarsa(Q_sarsa, sarsa_episodes, socket, steps=150)
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
        Q_ql, ql_episodes = ql.ql(Q_ql, ql_episodes, socket, steps=150)
        print("Iteration:", i, "|", "Failed Episodes:", len(ql_episodes))
        i += 1

    Q_new = [[[[0 for _ in range(n)] for _ in range(3)] for _ in range(n)] for _ in range(n)]
    with open('Q_ql.npy', 'wb') as f:
        for (t, w, x, v), _ in np.ndenumerate(np.array(Q_new)):
            Q_new[t][w][x][v] = int(Q_ql[t][w][x][v][0] < Q_ql[t][w][x][v][1])
        np.save(f, np.array(Q_new))


if __name__ == "__main__":
    th1 = Thread(target=main)
    th2 = Thread(target=run, daemon=True)

    th1.start()
    th2.start()
