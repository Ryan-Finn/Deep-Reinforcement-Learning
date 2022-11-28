import contextlib
from os import cpu_count

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from MountainCar import MountainCar
from SarsaLambda import SarsaLambda as sl

LAMBDA = 0.9
ALPHA = 0.001
GAMMA = 1.0
EPSILON = 0.0
EPISODES = 100
RUNS = max(cpu_count() - 1, 1)
MAX_STEPS = 1000


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def learn(order, grid_size, run):
    x, y, z, w = [], [], [], None
    model = MountainCar()
    steps = np.zeros(EPISODES)
    sarsa_lam = sl(model, LAMBDA, ALPHA, GAMMA, EPSILON, order, MAX_STEPS)
    positions = np.linspace(model.min_maxes[0], model.min_maxes[1], grid_size)
    velocities = np.linspace(model.min_maxes[2], model.min_maxes[3], grid_size)

    with tqdm(total=EPISODES, desc='Run: %d' % (run + 1), leave=False) as progress:
        for episode in range(EPISODES):
            steps[episode] = sarsa_lam.learnEpisode()
            progress.update()
        w = sarsa_lam.weights

    for position in positions:
        for velocity in velocities:
            model.set([position, velocity])
            x.append(position)
            y.append(velocity)
            z.append(sarsa_lam.cost_to_go())

    return steps, x, y, z, w


def main():
    grid_size = 40
    orders = [3]  # , 5, 7]
    steps = np.zeros((len(orders), EPISODES))
    x = np.zeros((len(orders), grid_size ** 2))
    y = np.zeros((len(orders), grid_size ** 2))
    z = np.zeros((len(orders), grid_size ** 2))
    w = sl(MountainCar(), LAMBDA, ALPHA, GAMMA, EPSILON, 3, MAX_STEPS).weights

    with joblib.Parallel(n_jobs=RUNS) as parallel:
        with tqdm_joblib(tqdm(desc="Fourier SARSA(Lambda)", total=len(orders) * RUNS, ncols=100)):
            for i in range(len(orders)):
                results = parallel(joblib.delayed(learn)(orders[i], grid_size, run) for run in range(RUNS))
                steps[i, :] = np.sum([r[0] for r in results], axis=0) / RUNS
                x[i, :] = np.sum([r[1] for r in results], axis=0) / RUNS
                y[i, :] = np.sum([r[2] for r in results], axis=0) / RUNS
                z[i, :] = np.sum([r[3] for r in results], axis=0) / RUNS
                w[i, :] = np.sum([r[4] for r in results], axis=0) / RUNS

    # Figure 1
    for i, order in enumerate(orders):
        plt.plot(steps[i], label='Order = %d' % order)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.yscale('log')
    plt.legend()
    plt.savefig('images/figure_1.png')
    plt.close()

    # Figures 2 - 4
    for i, order in enumerate(orders):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[i], y[i], z[i])
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Cost to go')
        ax.set_title('O(%d)\nEpisode %d' % (order, EPISODES))
        plt.savefig('images/figure_%d.png' % (i + 2))
        plt.close()


if __name__ == '__main__':
    main()
