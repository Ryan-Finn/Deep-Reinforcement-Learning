import struct
from threading import Thread
from numpy import pi

import zmq

from server import main as run


def main():
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
