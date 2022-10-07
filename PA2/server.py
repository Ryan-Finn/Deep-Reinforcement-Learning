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
        self.ax.set_ylim(-5, 5)

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
    w = spc.W / 2
    state = [0, 0, np.pi, 0]
    yield state

    while True:
        response_bytes = sock.recv()
        u, = struct.unpack('f', response_bytes[0:])
        new_state = solve_ivp(spc.deriv, [0, timestep], state, args=[[u], 0]).y[:, -1]

        # Elastic collision
        if new_state[0] >= 5 - w or new_state[0] <= -5 + w:
            new_state[0] = state[0]
            new_state[1] = -state[1]
            new_state[2] = state[2]
            new_state[3] = state[3] - 2 * state[1] * np.cos(state[2])

        state = new_state
        sock.send(struct.pack('ffff', *state))
        yield state


def main():
    # Pole mass, width, and height
    m = 1
    w = 0.1
    h = 1
    # Cart mass, width, and height
    M = 1
    W = 0.5
    H = 0.2
    # Gravity
    g = 9.81

    spc = ip.SinglePendulumCart(m, w, h, M, W, H, g)

    cont = zmq.Context()
    sock = cont.socket(zmq.REP)
    sock.bind("tcp://*:5556")

    vis = Visualizer(spc)
    _ = anim.FuncAnimation(vis.fig, vis.animate, producer(spc, sock, 0.01), vis.init_patches, interval=1, blit=True)
    plt.show()
