import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER PARAMETERS
# ============================================================

num_arms = 10
T = 250
MC_RUNS = int(input("Monte Carlo runs: "))

tau = float(input("Softmax temperature (tau): "))
alpha = float(input("Learning rate (alpha): "))

initial_resources = float(input("Initial resource endowment: "))
exploration_cost = float(input("Exploration cost: "))
pivot_cost = float(input("Pivot cost: "))
pivot_threshold = float(input("Pivot threshold (belief gap): "))

# ============================================================
# TRUE ENVIRONMENT
# ============================================================

# ============================================================
# TRUE ENVIRONMENT (SPACE-SEPARATED INPUT)
# ============================================================

true_means = np.fromstring(
    input(f"Enter true means (space-separated, {num_arms} values): "),
    sep=' '
)

true_vars = np.fromstring(
    input(f"Enter true variances (space-separated, {num_arms} values): "),
    sep=' '
)

assert len(true_means) == num_arms, "You must enter exactly num_arms values"
assert len(true_vars) == num_arms, "You must enter exactly num_arms values"

true_stds = np.sqrt(true_vars)


# ============================================================
# INITIAL BELIEFS
# ============================================================

# ============================================================
# INITIAL BELIEFS (SPACE-SEPARATED INPUT)
# ============================================================

Q0 = np.fromstring(
    input(f"Enter initial beliefs Q0 (space-separated, {num_arms} values): "),
    sep=' '
)

assert len(Q0) == num_arms, "You must enter exactly num_arms values"

# ============================================================
# FUNCTIONS
# ============================================================

def softmax(Q, tau):
    z = Q / tau
    z -= np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def true_policy(means):
    p = np.zeros_like(means)
    p[np.argmax(means)] = 1.0
    return p

P_true = true_policy(true_means)

# ============================================================
# STORAGE
# ============================================================

Q_paths = np.zeros((MC_RUNS, T, num_arms))
KL_paths = np.zeros((MC_RUNS, T))
pivot_paths = np.zeros((MC_RUNS, T))
reward_paths = np.zeros((MC_RUNS, T))
final_resources = np.zeros(MC_RUNS)

# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

for run in range(MC_RUNS):

    Q = Q0.copy()
    resources = initial_resources
    dominant_arm = np.argmax(Q)

    for t in range(T):

        if resources <= 0:
            break

        probs = softmax(Q, tau)
        arm = np.random.choice(num_arms, p=probs)

        reward = np.random.normal(true_means[arm], true_stds[arm])
        reward_paths[run, t] = reward

        # Learning
        Q[arm] += alpha * (reward - Q[arm])

        new_dominant = np.argmax(Q)

        # Pivot logic (belief-justified pivot)
        pivot_event = (
            arm == new_dominant and
            new_dominant != dominant_arm and
            Q[new_dominant] - Q[dominant_arm] > pivot_threshold
        )

        exploration_event = (arm != dominant_arm) and not pivot_event

        if pivot_event:
            pivot_paths[run, t] = 1
            resources -= pivot_cost
            dominant_arm = new_dominant

        elif exploration_event:
            resources -= exploration_cost

        Q_paths[run, t, :] = Q.copy()

        # Judgment quality
        P_belief = softmax(Q, tau)
        KL_paths[run, t] = kl_divergence(P_true, P_belief)

    final_resources[run] = resources

# ============================================================
# AGGREGATED OUTPUTS
# ============================================================

mean_KL = KL_paths.mean(axis=0)
mean_rewards = reward_paths.cumsum(axis=1).mean(axis=0)
mean_pivots = pivot_paths.cumsum(axis=1).mean(axis=0)

# ============================================================
# PLOTS
# ============================================================

plt.figure()
plt.plot(mean_KL)
plt.title("Mean KL Divergence (Belief vs Reality)")
plt.xlabel("Time")
plt.ylabel("KL")
plt.show()

plt.figure()
plt.plot(mean_rewards)
plt.title("Mean Cumulative Reward")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.show()

plt.figure()
plt.plot(mean_pivots)
plt.title("Mean Cumulative Pivots")
plt.xlabel("Time")
plt.ylabel("Pivots")
plt.show()

plt.figure()
plt.hist(final_resources, bins=20)
plt.title("Final Resource Distribution")
plt.xlabel("Resources")
plt.ylabel("Frequency")
plt.show()
