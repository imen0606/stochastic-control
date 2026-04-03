"""Visualize the parameter landscape of the Financial RLVR Gym.

Produces plots showing how optimal strategy, greedy performance,
and the planning gap vary across κ, λ, and T.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

with open("docs/data/parameter_landscape.json") as f:
    data = json.load(f)

kappas = sorted(set(r["kappa"] for r in data))
lambdas = sorted(set(r["lam"] for r in data))
Ts = sorted(set(r["T"] for r in data))


def get_val(T, kappa, lam, field):
    for r in data:
        if r["T"] == T and r["kappa"] == kappa and r["lam"] == lam:
            return r[field]
    return np.nan


# ============================================================
# FIGURE 1: Optimal Strategy — Mean Switches (how active is optimal?)
# ============================================================
fig, axes = plt.subplots(1, len(Ts), figsize=(5 * len(Ts), 4.5), sharey=True)
fig.suptitle("OPTIMAL STRATEGY: Mean Switches per Episode\n(How active is the Bellman-optimal policy?)",
             fontsize=14, fontweight="bold", y=1.02)

for idx, T in enumerate(Ts):
    ax = axes[idx]
    grid = np.array([[get_val(T, k, l, "opt_mean_sw") for l in lambdas] for k in kappas])
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(3, np.nanmax(grid)))
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.2f}" for l in lambdas], fontsize=8)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.1f}" for k in kappas], fontsize=8)
    ax.set_xlabel("λ (switching cost)")
    if idx == 0:
        ax.set_ylabel("κ (mean-reversion speed)")
    ax.set_title(f"T = {T}", fontsize=11)
    # Annotate
    for i in range(len(kappas)):
        for j in range(len(lambdas)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(grid) * 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("docs/plot_1_optimal_switches.png", dpi=150, bbox_inches="tight")
print("Saved plot_1_optimal_switches.png")


# ============================================================
# FIGURE 2: Greedy Per-Step Accuracy (how often does greedy match optimal?)
# ============================================================
fig, axes = plt.subplots(1, len(Ts), figsize=(5 * len(Ts), 4.5), sharey=True)
fig.suptitle("GREEDY PER-STEP ACCURACY vs OPTIMAL\n(What % of decisions does greedy get right?)",
             fontsize=14, fontweight="bold", y=1.02)

for idx, T in enumerate(Ts):
    ax = axes[idx]
    grid = np.array([[get_val(T, k, l, "gre_accuracy") * 100 for l in lambdas] for k in kappas])
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=80, vmax=100)
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.2f}" for l in lambdas], fontsize=8)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.1f}" for k in kappas], fontsize=8)
    ax.set_xlabel("λ (switching cost)")
    if idx == 0:
        ax.set_ylabel("κ (mean-reversion speed)")
    ax.set_title(f"T = {T}", fontsize=11)
    for i in range(len(kappas)):
        for j in range(len(lambdas)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 88 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.savefig("docs/plot_2_greedy_accuracy.png", dpi=150, bbox_inches="tight")
print("Saved plot_2_greedy_accuracy.png")


# ============================================================
# FIGURE 3: Greedy J Capture % (how much value does greedy capture?)
# ============================================================
fig, axes = plt.subplots(1, len(Ts), figsize=(5 * len(Ts), 4.5), sharey=True)
fig.suptitle("GREEDY J CAPTURE\n(What % of optimal profit does greedy earn?)",
             fontsize=14, fontweight="bold", y=1.02)

for idx, T in enumerate(Ts):
    ax = axes[idx]
    grid = np.array([[get_val(T, k, l, "j_capture") * 100 for l in lambdas] for k in kappas])
    # Clip for display
    grid = np.clip(grid, 0, 120)
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=20, vmax=105)
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.2f}" for l in lambdas], fontsize=8)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.1f}" for k in kappas], fontsize=8)
    ax.set_xlabel("λ (switching cost)")
    if idx == 0:
        ax.set_ylabel("κ (mean-reversion speed)")
    ax.set_title(f"T = {T}", fontsize=11)
    for i in range(len(kappas)):
        for j in range(len(lambdas)):
            v = grid[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}" if v < 120 else ">120"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if v < 60 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

plt.tight_layout()
plt.savefig("docs/plot_3_greedy_j_capture.png", dpi=150, bbox_inches="tight")
print("Saved plot_3_greedy_j_capture.png")


# ============================================================
# FIGURE 4: Combined "Gym Value" — low accuracy AND low J capture
# ============================================================
fig, axes = plt.subplots(1, len(Ts), figsize=(5 * len(Ts), 4.5), sharey=True)
fig.suptitle("GYM VALUE INDEX\n(100 - accuracy%) × (100 - J_capture%)\nHigher = more training signal + more room for improvement",
             fontsize=14, fontweight="bold", y=1.05)

for idx, T in enumerate(Ts):
    ax = axes[idx]
    grid = np.array([
        [(100 - get_val(T, k, l, "gre_accuracy") * 100) *
         max(0, 100 - min(get_val(T, k, l, "j_capture") * 100, 100))
         for l in lambdas]
        for k in kappas
    ])
    im = ax.imshow(grid, cmap="hot_r", aspect="auto", vmin=0, vmax=max(200, np.nanmax(grid)))
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.2f}" for l in lambdas], fontsize=8)
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.1f}" for k in kappas], fontsize=8)
    ax.set_xlabel("λ (switching cost)")
    if idx == 0:
        ax.set_ylabel("κ (mean-reversion speed)")
    ax.set_title(f"T = {T}", fontsize=11)
    for i in range(len(kappas)):
        for j in range(len(lambdas)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(grid) * 0.4 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("docs/plot_4_gym_value.png", dpi=150, bbox_inches="tight")
print("Saved plot_4_gym_value.png")


# ============================================================
# FIGURE 5: Summary scatter — accuracy vs J capture
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("ACCURACY vs J CAPTURE — Every Parameter Combination\n"
             "Bottom-left = hardest (many wrong decisions + high value loss)",
             fontsize=13, fontweight="bold")

markers = {5: "o", 10: "s", 15: "D", 25: "^"}
colors_kappa = plt.cm.viridis(np.linspace(0.1, 0.9, len(kappas)))

for r in data:
    acc = r["gre_accuracy"] * 100
    cap = min(r["j_capture"] * 100, 120)
    k_idx = kappas.index(r["kappa"])
    marker = markers[r["T"]]
    ax.scatter(acc, cap, c=[colors_kappa[k_idx]], marker=marker, s=60, alpha=0.8,
               edgecolors="black", linewidths=0.5)

# Legend for T
for T, m in markers.items():
    ax.scatter([], [], marker=m, c="gray", s=60, label=f"T={T}", edgecolors="black", linewidths=0.5)
# Legend for kappa
for i, k in enumerate(kappas):
    ax.scatter([], [], marker="o", c=[colors_kappa[i]], s=60, label=f"κ={k}", edgecolors="black", linewidths=0.5)

ax.set_xlabel("Greedy Per-Step Accuracy (%)", fontsize=12)
ax.set_ylabel("Greedy J Capture (%)", fontsize=12)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
ax.axvline(x=90, color="gray", linestyle="--", alpha=0.3)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.set_xlim(82, 100.5)
ax.set_ylim(20, 115)

# Annotate sweet spot
ax.annotate("SWEET SPOT\n(many wrong decisions\n+ high value loss)",
            xy=(85, 45), fontsize=10, fontweight="bold", color="red",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="red"))

plt.tight_layout()
plt.savefig("docs/plot_5_accuracy_vs_capture.png", dpi=150, bbox_inches="tight")
print("Saved plot_5_accuracy_vs_capture.png")

print("\nAll 5 plots saved to docs/")
