from os import remove
import struct
from threading import Thread

import numpy as np
import zmq

from server import main as run

n = 8  # Number of bins (even). Try to keep low, Q = O(n^5)
bins = [
    np.linspace(0, np.pi, n),  # theta bins
    np.linspace(0, 4, n),  # omega bins
    np.linspace(0, 5, n),  # x bins
    np.linspace(0, 7, n)  # velocity bins
]
u = np.linspace(-10, 10, n)  # force bins


def main():
    Q = np.array([[[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)])

    for x in range(n):
        for v in range(n):
            for a in range(n):
                Q[0][0][x][v][a] = 1

    APPLY_FORCE = 0
    SET_STATE = 1
    TRAINING = 2

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', TRAINING))
    _ = socket.recv()

    episode = 0
    for (t, w, x, v, a), val in np.ndenumerate(Q):
        state = [bins[0][t], bins[1][w], bins[2][x], bins[3][v]]
        socket.send(struct.pack('iffff', SET_STATE, *state))
        _ = socket.recv()

        socket.send(struct.pack('if', APPLY_FORCE, u[a]))
        tp, wp, xp, vp = struct.unpack('ffff', socket.recv())
        R = (-1 + abs(wp) / 4) * abs(tp) / np.pi - abs(wp) / 8
        tp, wp, xp, vp = discretize([tp, wp, xp, vp])

        maxQp = -np.inf
        for Ap in range(n):
            if maxQp < Q[tp][wp][xp][vp][Ap]:
                maxQp = Q[tp][wp][xp][vp][Ap]

        Q[t][w][x][v][a] += (R + 1 * maxQp - val) / 3000
        t, w, x, v = tp, wp, xp, vp

        if t == 0 and w == 0:
            Q[t][w][x][v][a] = 1
            episode += 1
            print("Episode:", episode, "|", "Step Count:", 1)
            continue

        count = 1
        for i in range(1, 2500):
            maxQ = -np.inf
            maxA = 0
            for A in range(n):
                if maxQ < Q[t][w][x][v][A]:
                    maxQ = Q[t][w][x][v][A]
                    maxA = A

            socket.send(struct.pack('if', APPLY_FORCE, u[maxA]))
            tp, wp, xp, vp = struct.unpack('ffff', socket.recv())
            R = (-1 + abs(wp) / 4) * abs(tp) / np.pi - abs(wp) / 8
            tp, wp, xp, vp = discretize([tp, wp, xp, vp])

            maxQp = -np.inf
            for Ap in range(n):
                if maxQp < Q[tp][wp][xp][vp][Ap]:
                    maxQp = Q[tp][wp][xp][vp][Ap]

            Q[t][w][x][v][maxA] += (R + 1 * maxQp - maxQ) / 3000
            t, w, x, v = tp, wp, xp, vp

            if t == 0 and w == 0:
                Q[t][w][x][v][maxA] = 1
                break
            count = i
        else:
            Q[t][w][x][v][a] = -1
            count = 2500

        episode += 1
        print("Episode:", episode, "|", "Step Count:", count)

    for t in range(n):
        for w in range(n):
            for x in range(n):
                for v in range(n):
                    QS = Q[t][w][x][v]
                    maxQ = -np.inf
                    maxA = 0

                    for A in range(n):
                        if maxQ < QS[A]:
                            maxQ = QS[A]
                            maxA = A

                    Q[t][w][x][v] = maxA

    remove('Q.npy')
    with open('Q.npy', 'wb') as f:
        np.save(f, Q)
    print(Q)


def discretize(state):
    for i, attr in enumerate(bins):
        for j in range(n):
            if abs(state[i]) < attr[j]:
                state[i] = j - 1
                break
            elif abs(state[i]) == attr[j]:
                state[i] = j
                break
        if abs(state[i]) > attr[n - 1]:
            state[i] = n - 1
        elif abs(state[i]) < attr[0]:
            state[i] = 0
    return state[0], state[1], state[2], state[3]


if __name__ == "__main__":
    th1 = Thread(target=main)
    th2 = Thread(target=run, daemon=True)

    th1.start()
    th2.start()
