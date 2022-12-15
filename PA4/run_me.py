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


def learn(animate):
    model_UP = CartPole()
    model_DOWN = CartPole()
    orders = [3]
    last = np.zeros(len(orders), dtype=int)
    steps = np.zeros((len(orders), MAX_EPISODES))

    for i, order in enumerate(orders):
        count = 0
        sarsa_lam_UP = sl(model_UP, LAMBDA, ALPHA, GAMMA, EPSILON, 0.0, -1.0, order, MAX_STEPS)
        sarsa_lam_DOWN = sl(model_DOWN, LAMBDA, ALPHA, GAMMA, EPSILON, 1.0, 0.0, order, MAX_STEPS)

        for episode in tqdm(range(MAX_EPISODES), desc=f'Learning Fourier SARSA(Lambda) O({order})', ncols=100):
            sarsa_lam_UP.newEpisode()
            sarsa_lam_DOWN.newEpisode()
            if animate:
                model_UP.animate(episode, 0, MAX_STEPS)
                model_DOWN.animate(episode, 0, MAX_STEPS)

            for step in range(MAX_STEPS):
                terminated = sarsa_lam_UP.step()
                if animate:
                    model_UP.animate(episode, step + 1, MAX_STEPS)

                if terminated:
                    steps[i, episode] = step + 1
                    break
            else:
                steps[i, episode] = MAX_STEPS

            # Count number of consecutive episodes the pole remains balanced for at least 95% of maximum number of steps
            count += 1 if steps[i, episode] >= 0.95 * MAX_STEPS else -count

            # If pole remains balanced consecutively for 20% of the episodes, it's safe to say convergence
            # was reached and there's no need to got through the rest of the episodes
            if episode > 20 and count > 0.2 * (episode + 1):
                print(f'Finished O({order})')
                last[i] = episode
                break
        else:  # If loop exited normally convergence was likely not reached, and it may fail to balance the pole well
            print(f'Max episode count ({MAX_EPISODES}) reached. Try retuning and/or rerunning.')
            continue  # Don't bother saving the weights in this case, they're probably garbage

        # If loop exited early then save the weights
        with open(f'weights/O({order})-Weights_UP', 'wb') as file:
            np.savez_compressed(file, sarsa_lam_UP.weights)

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


def play(order):
    weights = np.load(f'weights/O({order})-Weights')['arr_0']
    model = CartPole()
    # Having a small, non-zero epsilon could be a good way to simulate noise in the environment, if desired
    sarsa_lam = sl(model, epsilon=0.05, order=order, weights=weights)

    best_steps = 0
    episode = 1
    while True:
        sarsa_lam.newEpisode()
        model.animate(episode, 0, MAX_STEPS)

        steps = 0
        while True:
            terminated = sarsa_lam.step()
            model.animate(episode, steps + 1, MAX_STEPS)

            steps += 1
            if steps > best_steps:
                best_steps = steps

            if terminated:
                break

        episode += 1


if __name__ == '__main__':
    # learn(animate=False)
    play(order=3)
