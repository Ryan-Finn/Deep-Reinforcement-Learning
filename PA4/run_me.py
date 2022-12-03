import contextlib
from os import cpu_count

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from CartPole import CartPole
from SarsaLambda import SarsaLambda as sl

LAMBDA = 0.9
ALPHA = 0.005
GAMMA = 1.0
EPSILON = 0.01
ORDER = 3
EPISODES = 1000
RUNS = 1
MAX_STEPS = 1000
JOBS = max(min(cpu_count() - 1, RUNS), 1)


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


def learn(order, run):
    """Train SARSA(Lambda) and return steps per episode and weights"""
    model = CartPole()
    steps = np.zeros(EPISODES)
    sarsa_lam = sl(model, LAMBDA, ALPHA, GAMMA, EPSILON, order, MAX_STEPS)

    for episode in tqdm(range(EPISODES), desc=f'O({order}) Run: {run + 1}', leave=False):
        steps[episode] = sarsa_lam.learnEpisode()

    return steps, sarsa_lam.weights


def main():
    with joblib.Parallel(n_jobs=JOBS) as parallel:
        with tqdm_joblib(tqdm(desc="Learning Fourier SARSA(Lambda)", total=RUNS, ncols=100)):
            results = parallel(joblib.delayed(learn)(ORDER, run) for run in range(RUNS))
            steps = np.sum([r[0] for r in results], axis=0) / RUNS

            with open(f'weights/O({ORDER})-Weights', 'wb') as file:
                weights = np.sum([r[1] for r in results], axis=0) / RUNS
                np.savez_compressed(file, weights)

    # Figure 1
    plt.plot(steps, label=f'Order = {ORDER}')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()
    plt.savefig('images/figure_1.png')
    plt.close()

    # Figure 2
    plt.plot(steps, label=f'Order = {ORDER}')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.xscale('log')
    plt.legend()
    plt.savefig('images/figure_2.png')
    plt.close()


def animate():
    weights = np.load('weights/O(3)-Weights')['arr_0']
    sarsa_lam = sl(CartPole(), LAMBDA, ALPHA, GAMMA, EPSILON, ORDER, weights=weights)
    episodes = 1
    while True:
        sarsa_lam.playEpisode(episodes)
        episodes += 1


if __name__ == '__main__':
    # main()
    animate()
