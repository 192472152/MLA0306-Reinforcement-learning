import numpy as np

# ----------------------------------------------------
# Grid Setup
# ----------------------------------------------------
rows, cols = 5, 5                   # 5x5 city grid
goals = [(0, 4), (4, 4)]            # Delivery points
warehouse = (4, 0)                  # Start position

actions = ['U', 'D', 'L', 'R']
action_vectors = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

gamma = 0.9
theta = 1e-4                      # Threshold for stopping evaluation

# Reward map
rewards = np.zeros((rows, cols))
for g in goals:
    rewards[g] = 10                # Reward for reaching delivery points


# ----------------------------------------------------
# Helper: Take a step
# ----------------------------------------------------
def next_state(state, action):
    r, c = state
    dr, dc = action_vectors[action]
    nr, nc = r + dr, c + dc
    
    # Stay in boundaries
    if 0 <= nr < rows and 0 <= nc < cols:
        return (nr, nc)
    return (r, c)


# ----------------------------------------------------
# POLICY ITERATION
# ----------------------------------------------------
def policy_iteration():
    # Random initial policy
    policy = np.random.choice(actions, size=(rows, cols))
    V = np.zeros((rows, cols))

    stable = False
    iteration = 0

    while not stable:
        iteration += 1
        print(f"=== ITERATION {iteration} ===")
        
        # -------------------------
        # POLICY EVALUATION
        # -------------------------
        while True:
            delta = 0
            new_V = np.copy(V)
            
            for r in range(rows):
                for c in range(cols):
                    state = (r, c)
                    action = policy[r, c]
                    ns = next_state(state, action)
                    reward = rewards[ns]
                    
                    new_V[r, c] = reward + gamma * V[ns]
                    delta = max(delta, abs(new_V[r, c] - V[r, c]))
            
            V = new_V
            
            if delta < theta:
                break
        
        # -------------------------
        # POLICY IMPROVEMENT
        # -------------------------
        stable = True
        
        for r in range(rows):
            for c in range(cols):
                state = (r, c)
                
                # Compute best action
                action_returns = []
                for a in actions:
                    ns = next_state(state, a)
                    reward = rewards[ns]
                    action_returns.append(reward + gamma * V[ns])
                
                best_action = actions[np.argmax(action_returns)]
                
                # If policy changes, not stable
                if best_action != policy[r, c]:
                    stable = False
                
                policy[r, c] = best_action
    
    return policy, V


# ----------------------------------------------------
# Run Policy Iteration
# ----------------------------------------------------
optimal_policy, V = policy_iteration()

print("\nOptimal Value Function:")
print(V)

print("\nOptimal Policy (shortest path directions):")
print(optimal_policy)
