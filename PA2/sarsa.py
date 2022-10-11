import random
import struct

import bins

n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE = 0, [], 0, 0, 0, 0, 0


def init(a, b, c, d, e, f, g):
    global n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE
    n, u, epsilon, alpha, gamma, SET_STATE, APPLY_FORCE = a, b, c, d, e, f, g
    bins.init(n)


def sarsa(Q, episodes, socket, steps=150):
    episodes2 = []
    for (t0, w0, x0, v0) in episodes:
        state = bins.rand([t0, w0, x0, v0])
        socket.send(struct.pack('iffff', SET_STATE, *state))
        _ = socket.recv()

        A = e_greedy(Q, t0, w0, x0, v0)
        t, w, x, v = t0, w0, x0, v0
        for _ in range(0, steps):
            t, w, x, v, Q, A = updateQ(socket, Q, t, w, x, v, A)
            if (t == 0 or t == n - 1) and w == n // 2:
                break
        else:
            episodes2.append([t0, w0, x0, v0])

    return Q, episodes2


def updateQ(socket, Q, t, w, x, v, A):
    socket.send(struct.pack('if', APPLY_FORCE, u[A]))
    tp, wp, xp, vp = struct.unpack('ffff', socket.recv())

    tp, wp, xp, vp = bins.discretize([tp, wp, xp, vp])
    R = -1
    if (tp == 0 or tp == n - 1) and wp == n // 2:
        R = 0

    Ap = e_greedy(Q, tp, wp, xp, vp)

    Q[t][w][x][v][A] += alpha * (R + gamma * Q[tp][wp][xp][vp][Ap] - Q[t][w][x][v][A])
    return tp, wp, xp, vp, Q, Ap


def e_greedy(Q, t, w, x, v):
    if random.random() <= epsilon:
        return random.randint(0, 1)

    return int(Q[t][w][x][v][0] < Q[t][w][x][v][1])
