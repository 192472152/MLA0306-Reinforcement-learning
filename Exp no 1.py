import numpy as np
import random

GRID_SIZE = 5

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

grid = np.zeros((GRID_SIZE, GRID_SIZE))

dirt_positions = [(0,4), (2,1), (3,3), (4,0)]
for r,c in dirt_positions:
    grid[r][c] = 1

obstacle_positions = [(1,2), (2,3), (4,4)]
for r,c in obstacle_positions:
    grid[r][c] = -1

def step(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return (r, c), -1

    reward = grid[nr][nc]
    return (nr, nc), reward

def random_policy(state): # Modified to accept 'state' argument
    return random.choice([0,1,2,3])

def greedy_policy(state):
    r, c = state
    if c < GRID_SIZE - 1:
        return 3
    return 1

def spiral_policy(state):
    r, c = state
    if r <= c and r + c < GRID_SIZE - 1:
        return 3
    elif r < c and r + c >= GRID_SIZE - 1:
        return 1
    elif r >= c and r + c > GRID_SIZE - 1:
        return 2
    else:
        return 0

def run_simulation(policy_function, steps=50):
    state = (0, 0)
    total_reward = 0
    visited = []

    print(f"\nRunning Policy: {policy_function.__name__}")
    print("Starting at (0, 0)\n")

    for _ in range(steps):
        action = policy_function(state)
        next_state, reward = step(state, action)
        visited.append((state, action, reward))
        total_reward += reward
        state = next_state

    print("Visited States (state, action, reward):")
    for v in visited:
        print(v)

    print(f"\nTotal Reward Collected: {total_reward}")
    return total_reward

run_simulation(random_policy)
run_simulation(greedy_policy)
run_simulation(spiral_policy)
