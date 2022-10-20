import struct

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import zmq
from scipy.integrate import solve_ivp

import sim as ip


class Visualizer:
    def __init__(self, spc):
        self.spc = spc
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-3, 7)
        plt.axis('equal')

    def init_patches(self):
        patches = self.spc.draw(self.ax, [0, 0, 0, 0])

        for patch in patches:
            self.ax.add_patch(patch)

        return patches

    def animate(self, state):
        patches = self.spc.draw(self.ax, state)
        return patches


def producer(spc, sock, timestep):
    a = 0  # a = 0: Inelastic collision, a = 1: Elastic collision

    APPLY_FORCE = 0
    width = spc.W / 2

    t, w, x, v = struct.unpack('ffff', sock.recv()[4:])
    sock.send(struct.pack('i', 0))
    state = [t, w, x, v]
    yield state

    while True:
        response_bytes = sock.recv()

        if struct.unpack('i', response_bytes[0:4])[0] == APPLY_FORCE:
            u, = struct.unpack('f', response_bytes[4:])
            new_state = solve_ivp(spc.xddNtdd, [0, timestep], state, args=(u,)).y[:, -1]

            outbounds = 0
            if new_state[2] >= 5 - width:
                outbounds = 1
            elif new_state[2] <= -5 + width:
                outbounds = -1

            if outbounds != 0:
                new_state[1] -= state[3] * np.cos(new_state[0]) / spc.L
                new_state[2] = (1 + a) * outbounds * (5 - width) - a * new_state[2]
                new_state[3] = -a * state[3]

            state = new_state
            state[0] = (state[0] + np.pi) % (2 * np.pi) - np.pi
            sock.send(struct.pack('ffff', *state))
            yield state
        else:
            t, w, x, v = struct.unpack('ffff', response_bytes[4:])
            state = [t, w, x, v]
            sock.send(struct.pack('i', 0))


def main():
    RUNNING = 2

    # Pole mass, width, and height
    m = 0.25
    w = 0.2
    h = 4
    # Cart mass, width, and height
    M = 1
    W = 1
    H = 0.4
    # Gravity
    g = 9.81

    spc = ip.SinglePendulumCart(m, w, h, M, W, H, g)

    cont = zmq.Context()
    sock = cont.socket(zmq.REP)
    sock.bind("tcp://*:5556")

    animate = struct.unpack('i', sock.recv())[0] == RUNNING
    sock.send(struct.pack('i', 0))

    if animate:
        vis = Visualizer(spc)
        _ = anim.FuncAnimation(vis.fig, vis.animate, producer(spc, sock, 0.02), vis.init_patches, interval=1, blit=True)
        plt.show()
    else:
        for _ in producer(spc, sock, 0.02):
            pass
