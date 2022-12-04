import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from CartPole import CartPole
from SarsaLambda import SarsaLambda as sl

LAMBDA = 0.9
ALPHA = 0.001
GAMMA = 1.0
EPSILON = 0.05
ORDER = 3
MAX_EPISODES = 1000
MAX_STEPS = 500


def main():
    model = CartPole()
    steps = np.zeros(MAX_EPISODES)
    sarsa_lam = sl(model, LAMBDA, ALPHA, GAMMA, EPSILON, ORDER, MAX_STEPS)

    count = 0
    for episode in tqdm(range(MAX_EPISODES), desc=f'O({ORDER})', ncols=100):
        steps[episode] = sarsa_lam.learnEpisode()
        if steps[episode] >= 0.95 * MAX_STEPS:
            count += 1
        else:
            count = 0

        if count > 0.2 * (episode + 1):
            steps = steps[:episode]
            break

    with open(f'weights/O({ORDER})-Weights', 'wb') as file:
        np.savez_compressed(file, sarsa_lam.weights)

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
    weights = np.load(f'weights/O({ORDER})-Weights')['arr_0']
    sarsa_lam = sl(CartPole(), LAMBDA, ALPHA, GAMMA, 0.0, ORDER , weights=weights)
    episodes = 1
    while True:
        sarsa_lam.playEpisode(episodes)
        episodes += 1


if __name__ == '__main__':
    # main()
    animate()
