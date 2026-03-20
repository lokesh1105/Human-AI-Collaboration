import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# USER PARAMETERS


num_arms = 10
T = 100

MC_RUNS = int(input("Monte Carlo runs per type: "))

tau = float(input("Softmax temperature (tau > 0): "))
assert tau > 0

alpha = float(input("Human learning rate (alpha): "))

lambda_AI = float(input("Trust in AI (0–1): "))
assert 0 <= lambda_AI <= 1

initial_resources = float(input("Initial resource endowment: "))
exploration_cost = float(input("Exploration cost: "))
pivot_cost = float(input("Pivot cost: "))
pivot_threshold = float(input("Pivot threshold (belief gap): "))


# TRUE ENVIRONMENT


true_means = np.fromstring(
    input(f"Enter TRUE means ({num_arms} space-separated): "),
    sep=" "
)
true_vars = np.fromstring(
    input(f"Enter TRUE variances ({num_arms} space-separated): "),
    sep=" "
)

assert len(true_means) == num_arms
assert len(true_vars) == num_arms

true_stds = np.sqrt(true_vars)
optimal_arm = np.argmax(true_means)
optimal_reward = true_means[optimal_arm]


# USER-DEFINED INITIAL BELIEFS


Q0_novice = np.fromstring(
    input(f"Initial beliefs of NOVICES ({num_arms}): "),
    sep=" "
)
Q0_expert = np.fromstring(
    input(f"Initial beliefs of EXPERTS ({num_arms}): "),
    sep=" "
)
Q0_AI = np.fromstring(
    input(f"Initial beliefs of AI ({num_arms}): "),
    sep=" "
)

assert len(Q0_novice) == num_arms
assert len(Q0_expert) == num_arms
assert len(Q0_AI) == num_arms


# FUNCTIONS


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


# AGENT TYPES


TYPES = [
    "Novice",
    "Novice+AI",
    "Expert",
    "Expert+AI",
    "AI-only"
]

# ============================================================
# STORAGE
# ============================================================

reward_paths = defaultdict(list)
KL_paths = defaultdict(list)
regret_paths = defaultdict(list)
time_to_recognition = defaultdict(list)
pivot_quality = defaultdict(list)
net_reward_since_pivot = defaultdict(list)

# ============================================================
# MAIN SIMULATION
# ============================================================

for etype in TYPES:

    for run in range(MC_RUNS):

        # ---------------- INITIALIZATION ----------------

        if etype == "Novice":
            Q = Q0_novice.copy()
            human = True
            ai_assist = False
            ai_only = False

        elif etype == "Novice+AI":
            Q = Q0_novice.copy()
            human = True
            ai_assist = True
            ai_only = False

        elif etype == "Expert":
            Q = Q0_expert.copy()
            human = True
            ai_assist = False
            ai_only = False

        elif etype == "Expert+AI":
            Q = Q0_expert.copy()
            human = True
            ai_assist = True
            ai_only = False

        else:  # AI-only
            Q = None
            human = False
            ai_assist = True
            ai_only = True

        resources = initial_resources
        dominant_arm = np.argmax(Q) if human else None

        # AI posterior
        ai_mean = Q0_AI.copy()
        ai_var = np.ones(num_arms)

        rewards = np.zeros(T)
        KL = np.zeros(T)
        regret = np.zeros(T)
        net_since_pivot = np.zeros(T)

        t_recognition = -1
        current_net_reward = 0.0

        # ---------------- TIME LOOP ----------------

        for t in range(T):

            if resources <= 0:
                break

            # ----- ACTION SELECTION -----

            if ai_only:
                samples = np.random.normal(ai_mean, np.sqrt(ai_var))
                arm = np.argmax(samples)

            elif ai_assist:
                ai_sample = np.random.normal(ai_mean, np.sqrt(ai_var))
                Q_int = (1 - lambda_AI) * Q + lambda_AI * ai_sample
                probs = softmax(Q_int, tau)
                arm = np.random.choice(num_arms, p=probs)

            else:  # human only
                probs = softmax(Q, tau)
                arm = np.random.choice(num_arms, p=probs)

            # ----- REWARD -----

            reward = np.random.normal(true_means[arm], true_stds[arm])
            rewards[t] = reward
            regret[t] = max(0, optimal_reward - reward)

            # ----- UPDATE NET REWARD SINCE LAST PIVOT -----

            current_net_reward += reward

            # ----- HUMAN LEARNING -----

            if human:
                Q[arm] += alpha * (reward - Q[arm])

            # ----- AI BAYESIAN UPDATE -----

            prior_m = ai_mean[arm]
            prior_v = ai_var[arm]
            obs_v = true_vars[arm]

            post_v = 1 / (1 / prior_v + 1 / obs_v)
            post_m = post_v * (prior_m / prior_v + reward / obs_v)

            ai_mean[arm] = post_m
            ai_var[arm] = post_v

            # ----- RECOGNITION (HUMANS ONLY) -----

            if human and t_recognition == -1:
                if np.argmax(Q) == optimal_arm:
                    t_recognition = t

            # ----- PIVOT LOGIC (HUMANS ONLY) -----

            if human:
                new_dom = np.argmax(Q)

                pivot_event = (
                    arm == new_dom and
                    new_dom != dominant_arm and
                    Q[new_dom] - Q[dominant_arm] > pivot_threshold
                )

                if pivot_event:
                    pq = true_means[new_dom] - true_means[dominant_arm]
                    pivot_quality[etype].append(pq)

                    resources -= pivot_cost
                    dominant_arm = new_dom

                    current_net_reward = 0.0  # reset after pivot

                elif arm != dominant_arm:
                    resources -= exploration_cost
                    current_net_reward -= exploration_cost

                KL[t] = kl_divergence(P_true, softmax(Q, tau))

            net_since_pivot[t] = current_net_reward

        # ---------------- STORE RESULTS ----------------

        reward_paths[etype].append(rewards)
        KL_paths[etype].append(KL)
        regret_paths[etype].append(regret)
        time_to_recognition[etype].append(t_recognition)
        net_reward_since_pivot[etype].append(net_since_pivot)

# ============================================================
# PLOTS
# ============================================================

# Average reward
plt.figure()
for etype in TYPES:
    plt.plot(np.mean(reward_paths[etype], axis=0), label=etype)
plt.axhline(optimal_reward, linestyle="--", label="Optimal")
plt.title("Average Reward Over Time")
plt.legend()
plt.show()

# Cumulative regret
plt.figure()
for etype in TYPES:
    plt.plot(np.mean(regret_paths[etype], axis=0), label=etype)
plt.title("Cumulative Regret")
plt.legend()
plt.show()

# KL divergence (humans only)
plt.figure()
for etype in ["Novice", "Novice+AI", "Expert", "Expert+AI"]:
    plt.plot(np.mean(KL_paths[etype], axis=0), label=etype)
plt.title("Judgment Quality (KL Divergence)")
plt.legend()
plt.show()

# Pivot quality
plt.figure()
for etype in TYPES:
    if pivot_quality[etype]:
        plt.hist(pivot_quality[etype], bins=30, alpha=0.5, label=etype)
plt.axvline(0, linestyle="--")
plt.title("Pivot Quality Distribution")
plt.legend()
plt.show()

# Net reward since last pivot
plt.figure()
for etype in TYPES:
    plt.plot(np.mean(net_reward_since_pivot[etype], axis=0), label=etype)
plt.title("Net Reward Since Last Pivot")
plt.legend()
plt.show()
