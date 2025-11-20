import numpy as np
import matplotlib.pyplot as plt

true_ctr = np.array([0.05, 0.10, 0.20, 0.15])
n_ads = len(true_ctr)

def get_reward(ad):
    return 1 if np.random.rand() < true_ctr[ad] else 0

def epsilon_greedy(T=5000, epsilon=0.1):
    counts = np.zeros(n_ads)
    rewards = np.zeros(n_ads)
    total = []
    for t in range(T):
        if np.random.rand() < epsilon:
            ad = np.random.randint(n_ads)
        else:
            ad = np.argmax(rewards / (counts + 1e-7))
        r = get_reward(ad)
        counts[ad] += 1
        rewards[ad] += r
        total.append(r)
    return np.cumsum(total)

def ucb(T=5000):
    counts = np.zeros(n_ads)
    rewards = np.zeros(n_ads)
    total = []
    for ad in range(n_ads):
        r = get_reward(ad)
        counts[ad] += 1
        rewards[ad] += r
        total.append(r)
    for t in range(n_ads, T):
        ucb_values = (rewards / counts) + np.sqrt(2 * np.log(t) / counts)
        ad = np.argmax(ucb_values)
        r = get_reward(ad)
        counts[ad] += 1
        rewards[ad] += r
        total.append(r)
    return np.cumsum(total)

def thompson_sampling(T=5000):
    alpha = np.ones(n_ads)
    beta = np.ones(n_ads)
    total = []
    for t in range(T):
        samples = np.random.beta(alpha, beta)
        ad = np.argmax(samples)
        r = get_reward(ad)
        alpha[ad] += r
        beta[ad] += (1 - r)
        total.append(r)
    return np.cumsum(total)

T = 5000
eg = epsilon_greedy(T)
ucb = ucb(T)
ts = thompson_sampling(T)

plt.plot(eg, label="Epsilon-Greedy")
plt.plot(ucb, label="UCB")
plt.plot(ts, label="Thompson Sampling")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Clicks")
plt.title("Ad Selection Bandit Algorithm Comparison")
plt.legend()
plt.grid(True)
plt.show()
