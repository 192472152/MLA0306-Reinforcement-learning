import numpy as np

grid_size = 5

actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

item_positions = [(1,1), (3,2)]
goal_position = (4,4)
obstacle_positions = [(2,3)]
gamma = 0.9

def reward(state):
    if state in item_positions:
        return 2
    if state == goal_position:
        return 5
    if state in obstacle_positions:
        return -2
    return 0

def next_state(state, action):
    r, c = state
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size:
        return state
    return (nr, nc)

def fixed_policy(state):
    r, c = state
    if r < 4:
        return 1
    if c < 4:
        return 3
    return 1

states = [(r, c) for r in range(grid_size) for c in range(grid_size)]
V = {s: 0 for s in states}

def policy_evaluation(iterations=50):
    for _ in range(iterations):
        new_V = V.copy()
        for state in states:
            action = fixed_policy(state)
            ns = next_state(state, action)
            r = reward(ns)
            new_V[state] = r + gamma * V[ns]
        for s in states:
            V[s] = new_V[s]

policy_evaluation()

print("Value Function:")
for r in range(grid_size):
    row = []
    for c in range(grid_size):
        row.append(round(V[(r, c)], 2))
    print(row)
