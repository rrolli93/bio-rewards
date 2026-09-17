"""Visualize a discovery run.

Any search produced by ``optimize()`` carries a trajectory. ``plot_convergence``
turns that trajectory into the convergence curve, for any domain, in one call:

    from biorewards.optimize import optimize
    from biorewards.viz import plot_convergence
    res = optimize(reward_fn, seed, mutate)
    plot_convergence(res, "run.png", title="my binder search")

Matplotlib is imported lazily so the core library has no hard plotting dependency.
"""

from __future__ import annotations


def plot_convergence(result, path, title="reward convergence", subtitle=None):
    """Render the reward climb of an OptimizeResult to an image file.

    Plots best-so-far (the monotone climb) and the population mean (the
    exploration around it), marks the start and the converged endpoint, and
    returns the path written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_convergence needs matplotlib (pip install matplotlib)."
        ) from exc

    traj = result.trajectory
    it = [t["iteration"] for t in traj]
    best = [float(t["best_reward"]) for t in traj]
    mean = [float(t.get("mean_reward", t["best_reward"])) for t in traj]

    INK = "#0B0F17"; GOLD = "#F2B705"; MUT = "#8593ab"; GRID = "#24304a"
    RED = "#E1584C"; GREEN = "#35B27A"
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(INK); ax.set_facecolor(INK)
    ax.plot(it, mean, color=MUT, lw=1.3, alpha=0.8, label="population mean (exploration)")
    ax.plot(it, best, color=GOLD, lw=2.6, label="best so far (monotone)")
    ax.axhline(1.0, color=GREEN, lw=1, ls="--", alpha=0.5)
    ax.plot(it[0], best[0], "o", color=RED, ms=8, mec=INK, mew=1.4)
    ax.annotate(f"start  R={best[0]:.3f}", (it[0], best[0]),
                (it[0] + max(it) * 0.04, best[0]), color=RED, fontsize=11)
    ax.plot(it[-1], best[-1], "o", color=GREEN, ms=8, mec=INK, mew=1.4)
    ax.annotate(f"converged  R={best[-1]:.3f}", (it[-1], best[-1]),
                (it[-1] - max(it) * 0.45, best[-1] - 0.12), color=GREEN, fontsize=11)

    ax.set_ylim(0, 1.05); ax.set_xlabel("iteration"); ax.set_ylabel("reward  R  (0 to 1)")
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUT)
    ax.xaxis.label.set_color("#E6EAF2"); ax.yaxis.label.set_color("#E6EAF2")
    full = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full, color="#E6EAF2", fontsize=14, loc="left", pad=10)
    leg = ax.legend(facecolor="#131A26", edgecolor=GRID, labelcolor="#E6EAF2",
                    fontsize=10, loc="lower right")
    leg.get_frame().set_alpha(0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=INK)
    plt.close(fig)
    return path
