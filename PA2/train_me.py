import random
import struct
from threading import Thread

import numpy as np
import zmq

from server import main as run

n = 7  # Number of bins (odd). Try to keep low, Q = O(n^5)
bins = [
    np.linspace(-np.pi, np.pi, n),  # theta bins
    np.linspace(-4, 4, n),  # omega bins
    np.linspace(-4.5, 4.5, n),  # x bins
    np.linspace(-7, 7, n)  # velocity bins
]
u = np.linspace(-10, 10, n)  # force bins


def main():
    TRAINING = 2
    Q = np.array([[[[[-1 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)])

    for x in range(n):
        for v in range(n):
            for a in range(n):
                Q[n // 2][n // 2][x][v][a] = 0

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', TRAINING))
    _ = socket.recv()

    states = []
    for (t, w, x, v, a), val in np.ndenumerate(Q):
        states.append([t, w, x, v, a])

    for i in range(5):
        print("\nPass:", i + 1)
        Q, states = func(Q, states, socket, max_steps=500)

    Qnew = np.array([[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)])
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

                    Qnew[t][w][x][v] = maxA

    # remove('Q.npy')
    with open('Q.npy', 'wb') as f:
        np.save(f, Qnew)


def func(Q, states, socket, max_steps=500):
    APPLY_FORCE = 0
    SET_STATE = 1

    states2 = []
    episode = 0
    for (t, w, x, v, a) in states:
        state = rand([t, w, x, v])
        socket.send(struct.pack('iffff', SET_STATE, *state))
        _ = socket.recv()

        socket.send(struct.pack('if', APPLY_FORCE, u[a]))
        tp, wp, xp, vp = struct.unpack('ffff', socket.recv())
        R = (-1 + abs(wp) / 4) * abs(tp) / np.pi - abs(wp) / 8
        tp, wp, xp, vp = discretize([tp, wp, xp, vp])
        # R = -1
        # if tp == n // 2 and wp == n // 2:
        #     R = 0

        maxQp = -np.inf
        for Ap in range(n):
            if maxQp < Q[tp][wp][xp][vp][Ap]:
                maxQp = Q[tp][wp][xp][vp][Ap]

        Q[t][w][x][v][a] += (R + 1 * maxQp - Q[t][w][x][v][a]) / max_steps
        t, w, x, v = tp, wp, xp, vp

        if t == n // 2 and w == n // 2:
            Q[t][w][x][v][a] = 0
            episode += 1
            print("Episode:", episode, "|", "Step Count:", 1)
            continue

        count = 1
        for i in range(1, max_steps):
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
            # R = -1
            # if tp == n // 2 and wp == n // 2:
            #     R = 0

            maxQp = -np.inf
            for Ap in range(n):
                if maxQp < Q[tp][wp][xp][vp][Ap]:
                    maxQp = Q[tp][wp][xp][vp][Ap]

            Q[t][w][x][v][maxA] += (R + 1 * maxQp - maxQ) / max_steps
            t, w, x, v = tp, wp, xp, vp

            if t == n // 2 and w == n // 2:
                Q[t][w][x][v][maxA] = 0
                break
            count = i
        else:
            states2.append([t, w, x, v, a])
            Q[t][w][x][v][a] = -1
            count = max_steps

        episode += 1
        print("Episode:", episode, "|", "Step Count:", count)

    return Q, states2


def discretize(state):
    for i, attr in enumerate(bins):
        delta = (attr[n // 2] - attr[n // 2 - 1]) / 2
        if state[i] < attr[0] + delta:
            state[i] = 0
            continue

        for j in range(1, n):
            if state[i] < attr[j] - delta:
                state[i] = j - 1
                break
            elif state[i] == attr[j] - delta:
                state[i] = j
                break
        else:
            if state[i] > attr[n - 1] - delta:
                state[i] = n - 1

    return state[0], state[1], state[2], state[3]


def rand(state):
    for i, attr in enumerate(bins):
        delta = (attr[n // 2] - attr[n // 2 - 1]) / 2
        state[i] = random.uniform(attr[state[i]] - delta, attr[state[i]] + delta)

    return state[0], state[1], state[2], state[3]


if __name__ == "__main__":
    th1 = Thread(target=main)
    th2 = Thread(target=run, daemon=True)

    th1.start()
    th2.start()
