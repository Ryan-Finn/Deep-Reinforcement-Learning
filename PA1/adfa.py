import numpy as np

# initialize grid world
size = 4
goal = [3, 3]
teleport = [2, 0]
gamma = 0.95
theta = 1e-4

# left, up, right, down
actions = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]


# Define new policy
def stoc_policy(init=0.0):
    policy = np.empty((size, size), dtype=object)

    for i in range(size):
        for j in range(size):
            policy[i, j] = init * np.ones(size)

    return policy


# Iterate to get next state and reward
def iterate(state, action):
    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= size or y < 0 or y >= size:
        if state == goal:
            return goal, 0
        if state == teleport:
            return goal, -2.0
        return state, -1.0

    return next_state, -1.0


def main():
    policy = stoc_policy(0.25)

    # evaluate the policy
    v = np.zeros((size, size))
    while True:
        v_next = np.zeros((size, size))
        for (i, j), _ in np.ndenumerate(policy):
            for k, action in enumerate(actions):
                (i_next, j_next), reward = iterate([i, j], action)
                v_next[i, j] += policy[i, j][k] * (reward + gamma * v[i_next, j_next])
        if abs(v - v_next).max() < theta:
            v = v_next
            break
        v = v_next

    while True:
        # improve the policy
        policy = stoc_policy()
        deterministic_policy = stoc_policy()
        for (i, j), _ in np.ndenumerate(policy):
            best_a = -np.inf * np.ones(size)
            for k, action in enumerate(actions):
                (i_next, j_next), _ = iterate([i, j], action)
                best_a[k] = v[i_next, j_next]
            indices = np.where(best_a == max(best_a))[0]
            for ind in indices:
                policy[i, j][ind] = 1 / int(len(indices))
            deterministic_policy[i, j][indices[0]] = 1

        # evaluate the policy
        values = np.zeros((size, size))
        while True:
            v_next = np.zeros((size, size))
            for (i, j), _ in np.ndenumerate(policy):
                for k, action in enumerate(actions):
                    (i_next, j_next), reward = iterate([i, j], action)
                    v_next[i, j] += policy[i, j][k] * (reward + gamma * values[i_next, j_next])
            if abs(values - v_next).max() < theta:
                values = v_next
                break
            values = v_next

        if abs(v - values).max() < theta:
            v = values
            break

        v = values

    for input in deterministic_policy:
        for verd in input:
            print(*verd)

    print("\nv*")
    v = np.round(v, decimals=2)
    for i in range(size):
        val = ""
        for j in range(size):
            val += str(v[i, j]) + "  "
        print(val)

    print("\n")
    number = 1
    for input in policy:
        for verd in input:
            print(*verd)
            number *= int(len(np.where(verd != 0)[0]))

    print("\nNumber of Deterministic Policies:", number)


if __name__ == '__main__':
    main()
