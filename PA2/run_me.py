import struct
from threading import Thread
from numpy import pi

import zmq

from server import main as run
import numpy as np


def main():
    n = 9  # number of bins, should only be a power of 2 plus 1 (e.g. 3, 5, 9, 17, etc.)
    Q = [{} for _ in range(n)]
    Q = np.array([[[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)])
    actions = np.linspace(-5, 5, n)
    print(actions)

    APPLY_FORCE = 0
    SET_STATE = 1

    START = pi
    GOAL = 0

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('if', APPLY_FORCE, 500))
    socket.recv()

    while True:
        socket.send(struct.pack('if', APPLY_FORCE, 0))
        x, v, theta, omega = struct.unpack('ffff', socket.recv())


if __name__ == "__main__":
    th1 = Thread(target=main)
    th1.start()

    th2 = Thread(target=run)
    th2.start()
