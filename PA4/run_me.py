import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from CartPole import CartPole
from SarsaLambda import SarsaLambda as sl

LAMBDA = 0.9
ALPHA = 0.001
GAMMA = 1.0
EPSILON = 0.05  # Some exploration seems to be helpful
MAX_EPISODES = 1000
MAX_STEPS = 500


def main():
    model = CartPole()
    orders = [3, 5, 7]
    last = np.zeros(len(orders),  dtype=int)
    steps = np.zeros((len(orders), MAX_EPISODES))

    for i, order in enumerate(orders):
        count = 0
        sarsa_lam = sl(model, LAMBDA, ALPHA, GAMMA, EPSILON, order, MAX_STEPS)

        for episode in tqdm(range(MAX_EPISODES), desc=f'Learning Fourier SARSA(Lambda) O({order})', ncols=100):
            steps[i, episode] = sarsa_lam.learnEpisode()
            # Count number of consecutive episodes the pole remains balanced for at least 95% of maximum number of steps
            count += 1 if steps[i, episode] >= 0.95 * MAX_STEPS else -count

            # If pole remains balanced consecutively for at least 20% of the episodes, it's safe to say convergence
            # was reached and there's no need to got through the rest of the episodes
            if count > 0.2 * (episode + 1):
                print(f'Finished O({order})')
                last[i] = episode
                break
        else:  # If loop exited normally convergence was likely not reached, and it may fail to balance the pole well
            print(f'Max episode count ({MAX_EPISODES}) reached. Try retuning and/or rerunning.')
            continue  # Don't bother saving the weights in this case, they're probably garbage

        # If loop exited early then save the weights
        with open(f'weights/O({order})-Weights', 'wb') as file:
            np.savez_compressed(file, sarsa_lam.weights)

    # Figure 1
    for i, order in enumerate(orders):
        plt.plot(steps[i, :last[i]], label=f'Order = {order}')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()
    plt.savefig('images/figure_1.png')
    plt.close()

    # Figure 2
    for i, order in enumerate(orders):
        plt.plot(steps[i, :last[i]], label=f'Order = {order}')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.xscale('log')
    plt.legend()
    plt.savefig('images/figure_2.png')
    plt.close()


def animate(order):
    weights = np.load(f'weights/O({order})-Weights')['arr_0']
    sarsa_lam = sl(CartPole(), LAMBDA, ALPHA, GAMMA, 0.0, order, weights=weights)
    episodes = 1
    while True:
        sarsa_lam.playEpisode(episodes)
        episodes += 1


if __name__ == '__main__':
    # main()
    animate(order=5)
