"""3D surface visualizations of the gym parameter landscape.

Three axes: κ (signal persistence), λ (switching cost), T (horizon).
Colour represents metric value. One plot per metric.

Uses N=200 instances per parameter cell (pre-computed).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

with open("docs/parameter_landscape.json") as f:
    data = json.load(f)

kappas = sorted(set(r["kappa"] for r in data))
lambdas = sorted(set(r["lam"] for r in data))
Ts = sorted(set(r["T"] for r in data))


def get_val(T, kappa, lam, field):
    for r in data:
        if r["T"] == T and r["kappa"] == kappa and r["lam"] == lam:
            return r[field]
    return np.nan


def make_3d_plot(field, title, zlabel, cmap, vmin, vmax, filename, invert_z=False, multiply=100):
    """Create a 3D scatter/surface plot with κ, λ, T as axes and metric as colour."""
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Collect all points
    xs, ys, zs, cs = [], [], [], []
    for r in data:
        val = r[field]
        if multiply:
            val = val * multiply if field != "opt_mean_sw" else val
        # Clip extreme values
        val = np.clip(val, vmin, vmax)
        xs.append(r["kappa"])
        ys.append(r["lam"])
        zs.append(r["T"])
        cs.append(val)

    xs, ys, zs, cs = np.array(xs), np.array(ys), np.array(zs), np.array(cs)

    # Normalize colours
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cm.get_cmap(cmap)(norm(cs))

    # Size proportional to value (larger = more interesting)
    if invert_z:
        sizes = (vmax - cs) / (vmax - vmin) * 200 + 30
    else:
        sizes = (cs - vmin) / (vmax - vmin) * 200 + 30

    scatter = ax.scatter(xs, ys, zs, c=cs, cmap=cmap, s=sizes,
                          vmin=vmin, vmax=vmax, alpha=0.85,
                          edgecolors='black', linewidths=0.3)

    # Connect points at same κ,λ across T values (vertical lines)
    for kappa in kappas:
        for lam in lambdas:
            t_vals = []
            c_vals = []
            for T in Ts:
                val = get_val(T, kappa, lam, field)
                if multiply and field != "opt_mean_sw":
                    val *= multiply
                val = np.clip(val, vmin, vmax)
                t_vals.append(T)
                c_vals.append(val)
            ax.plot([kappa]*len(Ts), [lam]*len(Ts), t_vals,
                    color='gray', alpha=0.15, linewidth=0.5)

    ax.set_xlabel('κ (signal reversion speed)', fontsize=11, labelpad=10)
    ax.set_ylabel('λ (switching cost)', fontsize=11, labelpad=10)
    ax.set_zlabel('T (horizon)', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(zlabel, fontsize=11)

    ax.view_init(elev=25, azim=225)

    plt.tight_layout()
    plt.savefig(f"docs/{filename}", dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


def make_3d_surface_per_T(field, title, zlabel, cmap, vmin, vmax, filename, multiply=100):
    """Create a 3D surface for each T value: κ (x), λ (y), metric (z)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14),
                              subplot_kw={'projection': '3d'})
    fig.suptitle(f"{title}\n(N=200 instances per cell, α=0.30 fixed)",
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, T in enumerate(Ts):
        ax = axes[idx // 2][idx % 2]

        # Build surface grid
        K, L = np.meshgrid(kappas, lambdas, indexing='ij')
        Z = np.zeros_like(K)

        for i, kappa in enumerate(kappas):
            for j, lam in enumerate(lambdas):
                val = get_val(T, kappa, lam, field)
                if multiply and field != "opt_mean_sw":
                    val *= multiply
                Z[i, j] = np.clip(val, vmin, vmax)

        surf = ax.plot_surface(K, L, Z, cmap=cmap, alpha=0.85,
                                vmin=vmin, vmax=vmax,
                                edgecolors='black', linewidth=0.3)

        ax.set_xlabel('κ', fontsize=9, labelpad=5)
        ax.set_ylabel('λ', fontsize=9, labelpad=5)
        ax.set_zlabel(zlabel, fontsize=9, labelpad=5)
        ax.set_title(f'T = {T}', fontsize=12, fontweight='bold')
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=30, azim=225)
        ax.tick_params(labelsize=7)

    # Add one colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(zlabel, fontsize=11)

    plt.savefig(f"docs/{filename}", dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()


# ============================================================
# Plot A: Optimal Mean Switches — 3D surfaces per T
# ============================================================
make_3d_surface_per_T(
    field="opt_mean_sw",
    title="OPTIMAL STRATEGY: Mean Switches per Episode",
    zlabel="Switches",
    cmap="YlOrRd",
    vmin=0, vmax=4,
    filename="plot3d_1_optimal_switches.png",
    multiply=False,
)

# ============================================================
# Plot B: Greedy Accuracy — 3D surfaces per T
# ============================================================
make_3d_surface_per_T(
    field="gre_accuracy",
    title="GREEDY PER-STEP ACCURACY\n(What % of decisions does greedy get right?)",
    zlabel="Accuracy (%)",
    cmap="RdYlGn",
    vmin=80, vmax=100,
    filename="plot3d_2_greedy_accuracy.png",
)

# ============================================================
# Plot C: J Capture — 3D surfaces per T
# ============================================================
make_3d_surface_per_T(
    field="j_capture",
    title="GREEDY J CAPTURE\n(What % of optimal profit does greedy earn?)",
    zlabel="J Capture (%)",
    cmap="RdYlGn",
    vmin=20, vmax=105,
    filename="plot3d_3_j_capture.png",
)

# ============================================================
# Plot D: All parameters in one 3D scatter (κ, λ, T axes)
# ============================================================
make_3d_plot(
    field="gre_accuracy",
    title="GREEDY ACCURACY — Full Parameter Space\n(κ, λ, T axes; colour = accuracy %)",
    zlabel="Accuracy (%)",
    cmap="RdYlGn",
    vmin=82, vmax=100,
    filename="plot3d_4_accuracy_full.png",
    invert_z=True,
)

make_3d_plot(
    field="j_capture",
    title="GREEDY J CAPTURE — Full Parameter Space\n(κ, λ, T axes; colour = J capture %)",
    zlabel="J Capture (%)",
    cmap="RdYlGn",
    vmin=20, vmax=105,
    filename="plot3d_5_j_capture_full.png",
    invert_z=True,
)

# ============================================================
# Plot E: Gym Value — combined metric in full 3D
# ============================================================
# Compute gym value for each point
for r in data:
    acc = r["gre_accuracy"] * 100
    cap = min(r["j_capture"] * 100, 100)
    r["gym_value"] = (100 - acc) * max(0, 100 - cap)

make_3d_plot(
    field="gym_value",
    title="GYM VALUE INDEX — Full Parameter Space\n(Higher = more training signal + more room for improvement)",
    zlabel="Gym Value",
    cmap="hot_r",
    vmin=0, vmax=900,
    filename="plot3d_6_gym_value_full.png",
    multiply=False,
)

print("\nAll 3D plots saved to docs/")
