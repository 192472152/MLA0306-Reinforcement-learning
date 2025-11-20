import numpy as np

rows, cols = 5, 5
pickup_points = [(0, 4), (4, 2)]
gamma = 0.9
theta = 1e-4

actions = ['U', 'D', 'L', 'R']
action_move = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

rewards = np.full((rows, cols), -1.0)
for pr, pc in pickup_points:
    rewards[pr, pc] = 10

def next_state(state, action):
    r, c = state
    dr, dc = action_move[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        return (nr, nc)
    return (r, c)

def value_iteration():
    V = np.zeros((rows, cols))
    policy = np.full((rows, cols), 'U', dtype=str)

    while True:
        delta = 0
        new_V = np.copy(V)

        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                action_values = []
                for a in actions:
                    ns = next_state(state, a)
                    reward = rewards[ns]
                    action_values.append(reward + gamma * V[ns])
                new_V[r, c] = max(action_values)
                delta = max(delta, abs(new_V[r, c] - V[r, c]))

        V = new_V
        if delta < theta:
            break

    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            action_values = []
            for a in actions:
                ns = next_state(state, a)
                reward = rewards[ns]
                action_values.append(reward + gamma * V[ns])
            best_action = actions[np.argmax(action_values)]
            policy[r, c] = best_action

    return V, policy

V, policy = value_iteration()

print("\nOptimal Value Function:")
print(V)
print("\nOptimal Policy:")
print(policy)
