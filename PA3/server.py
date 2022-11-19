import struct

import zmq

import MountainCar


def producer(mountain_car, sock):
    SET_STATE = 1

    x, v = struct.unpack('ff', sock.recv()[4:])
    sock.send(struct.pack('i', 0))
    state = [x, v]
    yield state

    while True:
        response_bytes = sock.recv()

        if struct.unpack('i', response_bytes[0:4])[0] == SET_STATE:
            x, v = struct.unpack('ff', response_bytes[4:])
            mountain_car.set(x, v)
            sock.send(struct.pack('i', 0))
        else:
            action, = struct.unpack('f', response_bytes[4:])
            state = mountain_car.update(action)
            sock.send(struct.pack('ff', *state))
            yield state


def main():
    mountain_car = sim.MountainCar()

    cont = zmq.Context()
    sock = cont.socket(zmq.REP)
    sock.bind("tcp://*:5556")

    sock.send(struct.pack('i', 0))

    for _ in producer(mountain_car, sock):
        pass
