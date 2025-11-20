import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Price options (arms) and true conversion probabilities
# ----------------------------------------------------
prices = np.array([50, 70, 100])  
true_conversion = np.array([0.30, 0.20, 0.10])   # unknown to agent

# True expected revenue for reference
true_revenue = prices * true_conversion
print("True expected revenue per arm:", true_revenue)

# ----------------------------------------------------
# Helper: Generate reward (sale = 1) based on probability
# ----------------------------------------------------
def get_reward(arm):
    sale = np.random.rand() < true_conversion[arm]
    return prices[arm] if sale else 0


# ====================================================
# 1. EPSILON-GREEDY
# ====================================================
def epsilon_greedy(T=5000, epsilon=0.1):
    n = len(prices)
    rewards = np.zeros(n)
    counts = np.zeros(n)
    total_rewards = []

    for t in range(T):
        if np.random.rand() < epsilon:
            arm = np.random.randint(n)
        else:
            arm = np.argmax(rewards / (counts + 1e-7))

        reward = get_reward(arm)
        counts[arm] += 1
        rewards[arm] += reward
        total_rewards.append(reward)

    return np.cumsum(total_rewards)

# ====================================================
# 2. UCB (Upper Confidence Bound)
# ====================================================
def ucb(T=5000):
    n = len(prices)
    rewards = np.zeros(n)
    counts = np.zeros(n)
    total_rewards = []

    for t in range(n):
        reward = get_reward(t)
        rewards[t] += reward
        counts[t] += 1
        total_rewards.append(reward)

    for t in range(n, T):
        ucb_values = (rewards / counts) + np.sqrt(2 * np.log(t) / counts)
        arm = np.argmax(ucb_values)

        reward = get_reward(arm)
        rewards[arm] += reward
        counts[arm] += 1
        total_rewards.append(reward)

    return np.cumsum(total_rewards)

# ====================================================
# 3. THOMPSON SAMPLING
# ====================================================
def thompson_sampling(T=5000):
    n = len(prices)
    alpha = np.ones(n)
    beta = np.ones(n)
    total_rewards = []

    for t in range(T):
        samples = np.random.beta(alpha, beta)
        arm = np.argmax(samples)

        reward = get_reward(arm)
        sale = 1 if reward > 0 else 0

        alpha[arm] += sale
        beta[arm] += (1 - sale)

        total_rewards.append(reward)

    return np.cumsum(total_rewards)


# ----------------------------------------------------
# RUN ALL STRATEGIES
# ----------------------------------------------------
T = 5000
eg_rewards = epsilon_greedy(T)
ucb_rewards = ucb(T)
ts_rewards = thompson_sampling(T)

# ----------------------------------------------------
# PLOT RESULTS
# ----------------------------------------------------
plt.plot(eg_rewards, label="Epsilon-Greedy")
plt.plot(ucb_rewards, label="UCB")
plt.plot(ts_rewards, label="Thompson Sampling")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Revenue")
plt.title("Pricing Strategy Comparison (Multi-Armed Bandit)")
plt.legend()
plt.grid(True)
plt.show()
