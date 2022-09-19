#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')
ACTIONS_FIGS = ['←', '↑', '→', '↓']


def draw_V(v, INIT):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = v.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(v):
        # add state labels
        if [i, j] == INIT["goal"]:
            val = str(val) + "\nGOAL"
        if [i, j] == INIT["start"]:
            val = str(val) + "\nSTART"
        if [i, j] == INIT["tele"]:
            val = str(val) + "\nTELEPORT"

        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

        # Row and column labels...
    for i in range(len(v)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)


def draw_Pi(pi, INIT):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = pi.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), actions in np.ndenumerate(pi):
        val = ''

        for action in np.where(actions != 0)[0]:
            val += ACTIONS_FIGS[action]

        # add state labels
        if [i, j] == INIT["goal"]:
            val = str(val) + "\nGOAL"
        if [i, j] == INIT["start"]:
            val = str(val) + "\nSTART"
        if [i, j] == INIT["tele"]:
            val = str(val) + "\nTELEPORT"

        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(pi)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)
