"""
Created on Tue Dec 14 15:04:22 2021

@author: jonathan
"""
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np


class SinglePendulumCart:
    def __init__(self, m, w, h, M, W, H, g):
        self.m = m
        self.L = h
        self.M = M
        self.W = W
        self.g = g
        self.Pole = patches.Rectangle((-w / 2, 0), w, h, color='k')
        self.Cart = patches.Rectangle((-W / 2, -H / 2), W, H)

    def deriv(self, _, X, u, __):
        m = self.m
        L = self.L
        M = self.M
        g = self.g

        theta = X[0]
        omega = X[1]
        v = X[3]
        I = m * L * L / 3

        try:
            u = u[0]
        except IndexError:
            u = u

        A = np.array([
            [-(m + M), m * L * np.cos(theta)],
            [-np.cos(theta), L + I / m / L]
        ])

        b = np.array([
            [m * L * np.sin(theta) * omega * omega - u],
            [g * np.sin(theta)]
        ])

        a, omega_dot = np.linalg.solve(A, b)
        a = a[0]
        omega_dot = omega_dot[0]

        return np.array([omega, omega_dot, v, a])

    def draw(self, ax, y):
        x = y[2]
        theta = y[0]

        t1 = mpl.transforms.Affine2D().rotate(theta) + mpl.transforms.Affine2D().translate(x, 0) + ax.transData
        t2 = mpl.transforms.Affine2D().translate(x, 0) + ax.transData

        self.Pole.set_transform(t1)
        self.Cart.set_transform(t2)

        return self.Pole, self.Cart
