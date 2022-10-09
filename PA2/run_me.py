import struct
from threading import Thread

import zmq

from server import main as run
import numpy as np


def main():
    t = np.linspace(0, 9, 9)
    w = np.linspace(-7, 7, 9)
    x = np.linspace(-5, 5, 11)
    v = np.linspace(-7, 7, 9)
    actions = np.linspace(-5, 5, 9)

    keys = []
    for a in x:
        str1 = "" + str(a)
        for b in v:
            keys.append(str1 + str(b))

    Q = [[{key: [0 for _ in range(9)] for key in keys} for _ in range(9)] for _ in range(9)]  # Q(t, w, x, v, a)

    APPLY_FORCE = 0
    SET_STATE = 1

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    for episode in range(1000):
        state = [np.pi, 0, 0, 0]
        socket.send(struct.pack('iffff', SET_STATE, *state))
        socket.recv()

        while state[0] != 0 or state[1] != 0:
            socket.send(struct.pack('if', APPLY_FORCE, 5))
            # socket.recv()
            # socket.send(struct.pack('if', APPLY_FORCE, 0))
            t, w, x, v = struct.unpack('ffff', socket.recv())
            state = [t, w, x, v]


if __name__ == "__main__":
    th1 = Thread(target=main)
    th1.start()

    th2 = Thread(target=run)
    th2.start()
