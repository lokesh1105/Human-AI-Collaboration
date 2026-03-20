import numpy as np
import matplotlib.pyplot as plt
import secrets
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# USER PARAMETERS
# ============================================================

num_arms = 10
T = 100

MC_RUNS = int(input("Monte Carlo runs per type: "))

initial_resources = float(input("Initial resource endowment: "))
exploration_cost = float(input("Exploration cost: "))
pivot_cost = float(input("Pivot cost: "))
pivot_threshold = float(input("Pivot threshold (belief gap): "))

# ============================================================
# LEARNING & EXPLORATION GRID
# ============================================================

alpha_base = float(input("Enter base learning rate (alpha ≤ 0.4): "))

if alpha_base > 0.4:
    raise ValueError("alpha_base must be ≤ 0.4")

alpha_vals = np.round(np.arange(alpha_base, 0.401, 0.05), 2)
tau_vals   = np.round(np.arange(0.1, 2.01, 0.1), 2)

print("\nLearning rates:", alpha_vals)
print("Exploration rates:", tau_vals)

# ============================================================
# TRUE ENVIRONMENT
# ============================================================

true_means = np.fromstring(
    input(f"Enter TRUE means ({num_arms}): "), sep=" "
)
true_vars = np.fromstring(
    input(f"Enter TRUE variances ({num_arms}): "), sep=" "
)
true_stds = np.sqrt(true_vars)

# ============================================================
# INITIAL BELIEFS
# ============================================================

Q0_novice = np.fromstring(input("Initial beliefs of NOVICES: "), sep=" ")
Q0_expert = np.fromstring(input("Initial beliefs of EXPERTS: "), sep=" ")

# ============================================================
# FUNCTIONS
# ============================================================

def softmax(Q, tau):
    z = Q / tau
    z -= np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def tau_log_cooling(tau0, t):
    return tau0 / np.log(t + 2)

# ============================================================
# STORAGE
# ============================================================

Z = {
    "Novice": np.zeros((len(alpha_vals), len(tau_vals))),
    "Expert": np.zeros((len(alpha_vals), len(tau_vals)))
}

# ============================================================
# MAIN SIMULATION
# ============================================================

for etype in ["Novice", "Expert"]:

    for i, alpha in enumerate(alpha_vals):
        for j, tau0 in enumerate(tau_vals):

            mc_net_rewards = []

            for _ in range(MC_RUNS):
                np.random.seed(secrets.randbits(32))

                Q = Q0_novice.copy() if etype == "Novice" else Q0_expert.copy()
                resources = initial_resources
                dominant_arm = np.argmax(Q)

                net_cum_reward = 0.0

                for t in range(T):

                    if resources <= 0:
                        break

                    tau_t = max(0.01, tau_log_cooling(tau0, t))
                    arm = np.random.choice(num_arms, p=softmax(Q, tau_t))

                    reward = np.random.normal(true_means[arm], true_stds[arm])
                    net_cum_reward += reward

                    Q[arm] += alpha * (reward - Q[arm])

                    new_dom = np.argmax(Q)
                    if (
                        new_dom != dominant_arm and
                        Q[new_dom] - Q[dominant_arm] > pivot_threshold * tau_t
                    ):
                        resources -= pivot_cost
                        net_cum_reward -= pivot_cost
                        dominant_arm = new_dom
                    elif arm != dominant_arm:
                        resources -= exploration_cost
                        net_cum_reward -= exploration_cost

                mc_net_rewards.append(net_cum_reward)

            Z[etype][i, j] = np.mean(mc_net_rewards)

# ============================================================
# HEATMAPS
# ============================================================

for etype in ["Novice", "Expert"]:

    plt.figure(figsize=(9, 6))
    plt.imshow(
        Z[etype],
        origin="lower",
        aspect="auto",
        extent=[
            tau_vals[0], tau_vals[-1],
            alpha_vals[0], alpha_vals[-1]
        ]
    )
    plt.colorbar(label="Expected Net Cumulative Reward")
    plt.xlabel("Exploration (τ)")
    plt.ylabel("Learning rate (α)")
    plt.title(f"{etype}: Net Performance Heatmap")
    plt.tight_layout()
    plt.show()

# ============================================================
# 3D SURFACES
# ============================================================

Tau, Alpha = np.meshgrid(tau_vals, alpha_vals)

for etype in ["Novice", "Expert"]:

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Tau, Alpha, Z[etype],
        cmap="viridis",
        edgecolor="none"
    )

    ax.set_xlabel("Exploration (τ)")
    ax.set_ylabel("Learning rate (α)")
    ax.set_zlabel("Expected Net Cumulative Reward")
    ax.set_title(f"{etype}: Net Performance Surface")

    plt.tight_layout()
    plt.show()
