import struct
from threading import Thread

import zmq

from server import main as run


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    socket.send(struct.pack('f', 500))
    socket.recv()

    while True:
        socket.send(struct.pack('f', 0))
        response_bytes = socket.recv()

        x, _, _, _ = struct.unpack('ffff', response_bytes)


if __name__ == "__main__":
    th1 = Thread(target=main)
    th1.start()

    th2 = Thread(target=run)
    th2.start()
