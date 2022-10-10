import struct
from threading import Thread

import numpy as np
import zmq

from server import main as run

n = 9  # Number of bins (odd). Try to keep low: Q = O(n^2)
bins = [
    np.array([-np.pi / 18, 0, np.pi / 18]),  # theta bins
    np.linspace(-2, 2, n),  # omega bins
    np.linspace(-4.5, 4.5, 3),  # x bins
    np.linspace(-2, 2, n)  # velocity bins
]
u = [-10, 10]  # force bins

APPLY_FORCE = 0
SET_STATE = 1
RUNNING = 2


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', RUNNING))
    _ = socket.recv()

    with open('Q_ql.npy', 'rb') as f:
        Q = np.load(f)

    state = [0, 0, 0, 0]
    socket.send(struct.pack('iffff', SET_STATE, *state))
    _ = socket.recv()

    t, w, x, v = discretize(state)

    while True:
        A = Q[t][w][x][v]
        socket.send(struct.pack('if', APPLY_FORCE, u[A]))
        t, w, x, v = struct.unpack('ffff', socket.recv())
        t, w, x, v = discretize([t, w, x, v])


def discretize(state):
    if state[0] <= bins[0][0]:
        state[0] = 0
    elif state[0] >= bins[0][2]:
        state[0] = 2
    else:
        state[0] = 1

    delta = bins[2][2] / 2
    if state[2] <= bins[2][0] + delta:
        state[2] = 0
    elif state[2] >= bins[2][2] - delta:
        state[2] = 2
    else:
        state[2] = 1

    for i in [1, 3]:
        attr = bins[i]
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
    print((-np.pi) % (2 * np.pi))
    print((np.pi) % (2 * np.pi))

    # th1 = Thread(target=main, daemon=True)
    # th2 = Thread(target=run)
    #
    # th1.start()
    # th2.start()
