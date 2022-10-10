import struct
from threading import Thread

import numpy as np
import zmq

from server import main as run

n = 7  # Number of bins (odd). Try to keep low, Q = O(n^5)
bins = [
    np.linspace(-np.pi, np.pi, n),  # theta bins
    np.linspace(-4, 4, n),  # omega bins
    np.linspace(-5, 5, n),  # x bins
    np.linspace(-7, 7, n)  # velocity bins
]
u = np.linspace(-10, 10, n)  # force bins


def main():
    APPLY_FORCE = 0
    SET_STATE = 1
    RUNNING = 3

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', RUNNING))
    _ = socket.recv()

    with open('Q.npy', 'rb') as f:
        Q = np.load(f)

    state = [np.pi, 0, 0, 0]
    socket.send(struct.pack('iffff', SET_STATE, *state))
    _ = socket.recv()

    t, w, x, v = discretize(state)

    while True:
        A = Q[t][w][x][v]
        socket.send(struct.pack('if', APPLY_FORCE, u[A]))
        t, w, x, v = struct.unpack('ffff', socket.recv())
        t, w, x, v = discretize([t, w, x, v])


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


if __name__ == "__main__":
    th1 = Thread(target=main, daemon=True)
    th2 = Thread(target=run)

    th1.start()
    th2.start()
