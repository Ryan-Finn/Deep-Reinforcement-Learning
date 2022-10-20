import struct
from threading import Thread

import numpy as np
import zmq

import bins
from server import main as run
from train import n, u, SET_STATE, APPLY_FORCE

RUNNING = 2


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('i', RUNNING))
    _ = socket.recv()

    with open('policy.npy', 'rb') as f:
        Q = np.load(f)

    state = [0, 0, 0, 0]
    socket.send(struct.pack('iffff', SET_STATE, *state))
    _ = socket.recv()

    bins.init(n)
    t, w, x, v = bins.discretize(state)

    while True:
        A = Q[t][w][x][v]
        socket.send(struct.pack('if', APPLY_FORCE, u[A]))
        t, w, x, v = struct.unpack('ffff', socket.recv())
        t, w, x, v = bins.discretize([t, w, x, v])


if __name__ == "__main__":
    th1 = Thread(target=main, daemon=True)
    th2 = Thread(target=run)

    th1.start()
    th2.start()
