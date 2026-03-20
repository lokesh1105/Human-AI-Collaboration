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

# ============================================================
# TRUE ENVIRONMENT
# ============================================================

true_means = np.fromstring(input(f"Enter TRUE means ({num_arms}): "), sep=" ")
true_vars  = np.fromstring(input(f"Enter TRUE variances ({num_arms}): "), sep=" ")
true_stds  = np.sqrt(true_vars)

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

Z_cum = {
    "Novice": np.zeros((len(alpha_vals), len(tau_vals))),
    "Expert": np.zeros((len(alpha_vals), len(tau_vals)))
}

Z_net = {
    "Novice": np.zeros((len(alpha_vals), len(tau_vals))),
    "Expert": np.zeros((len(alpha_vals), len(tau_vals)))
}

reward_time = {"Novice": [], "Expert": []}

# ============================================================
# MAIN SIMULATION
# ============================================================

for etype in ["Novice", "Expert"]:

    Q0 = Q0_novice if etype == "Novice" else Q0_expert

    for i, alpha in enumerate(alpha_vals):
        for j, tau0 in enumerate(tau_vals):

            mc_cum = []
            mc_net = []
            mc_paths = []

            for _ in range(MC_RUNS):
                np.random.seed(secrets.randbits(32))

                Q = Q0.copy()
                resources = initial_resources
                dominant_arm = np.argmax(Q)

                cum_reward = 0.0
                net_reward = 0.0
                rewards = []

                for t in range(T):

                    if resources <= 0:
                        break

                    tau_t = max(0.01, tau_log_cooling(tau0, t))
                    arm = np.random.choice(num_arms, p=softmax(Q, tau_t))

                    reward = np.random.normal(true_means[arm], true_stds[arm])
                    rewards.append(reward)

                    Q[arm] += alpha * (reward - Q[arm])

                    cum_reward += reward
                    net_reward += reward

                    new_dom = np.argmax(Q)
                    if (
                        new_dom != dominant_arm and
                        Q[new_dom] - Q[dominant_arm] > pivot_threshold * tau_t
                    ):
                        resources -= pivot_cost
                        net_reward -= pivot_cost
                        dominant_arm = new_dom
                    elif arm != dominant_arm:
                        resources -= exploration_cost
                        net_reward -= exploration_cost

                mc_cum.append(cum_reward)
                mc_net.append(net_reward)
                mc_paths.append(rewards)

            Z_cum[etype][i, j] = np.mean(mc_cum)
            Z_net[etype][i, j] = np.mean(mc_net)

            if i == 0 and j == 0:
                max_len = max(len(r) for r in mc_paths)
                padded = np.full((MC_RUNS, max_len), np.nan)
                for k, r in enumerate(mc_paths):
                    padded[k, :len(r)] = r
                reward_time[etype] = np.nanmean(padded, axis=0)

# ============================================================
# TIME-SERIES PLOTS
# ============================================================

plt.figure()
plt.plot(reward_time["Novice"], label="Novices")
plt.plot(reward_time["Expert"], label="Experts")
plt.title("Average Reward Over Time")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.nancumsum(reward_time["Novice"]), label="Novices")
plt.plot(np.nancumsum(reward_time["Expert"]), label="Experts")
plt.title("Cumulative Reward Over Time")
plt.xlabel("Time")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()

# ============================================================
# MARGINAL PERFORMANCE CURVES
# ============================================================

plt.figure()
plt.plot(alpha_vals, np.mean(Z_cum["Novice"], axis=1), label="Novices")
plt.plot(alpha_vals, np.mean(Z_cum["Expert"], axis=1), label="Experts")
plt.title("Cumulative Reward vs Learning Rate (α)")
plt.xlabel("α")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()

plt.figure()
plt.plot(tau_vals, np.mean(Z_cum["Novice"], axis=0), label="Novices")
plt.plot(tau_vals, np.mean(Z_cum["Expert"], axis=0), label="Experts")
plt.title("Cumulative Reward vs Exploration (τ)")
plt.xlabel("τ")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.show()

plt.figure()
plt.plot(alpha_vals, np.mean(Z_net["Novice"], axis=1), label="Novices")
plt.plot(alpha_vals, np.mean(Z_net["Expert"], axis=1), label="Experts")
plt.title("Net Reward vs Learning Rate (α)")
plt.xlabel("α")
plt.ylabel("Net Reward")
plt.legend()
plt.show()

plt.figure()
plt.plot(tau_vals, np.mean(Z_net["Novice"], axis=0), label="Novices")
plt.plot(tau_vals, np.mean(Z_net["Expert"], axis=0), label="Experts")
plt.title("Net Reward vs Exploration (τ)")
plt.xlabel("τ")
plt.ylabel("Net Reward")
plt.legend()
plt.show()

# ============================================================
# HEATMAPS & 3D SURFACES
# ============================================================

Tau, Alpha = np.meshgrid(tau_vals, alpha_vals)

for etype in ["Novice", "Expert"]:
    for label, Z in [("Cumulative", Z_cum), ("Net", Z_net)]:

        plt.figure(figsize=(8,6))
        plt.imshow(Z[etype], origin="lower", aspect="auto",
                   extent=[tau_vals[0], tau_vals[-1],
                           alpha_vals[0], alpha_vals[-1]])
        plt.colorbar(label=f"{label} Reward")
        plt.xlabel("τ")
        plt.ylabel("α")
        plt.title(f"{etype}: {label} Reward Heatmap")
        plt.show()

        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Tau, Alpha, Z[etype], cmap="viridis")
        ax.set_xlabel("τ")
        ax.set_ylabel("α")
        ax.set_zlabel(f"{label} Reward")
        ax.set_title(f"{etype}: {label} Reward Surface")
        plt.show()
