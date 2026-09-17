"""Draw the mathematics: the reward as a landscape, and the search converging on it.

Produces (into deck/figures/):
  landscape_primer.png   the reward function R(Tm_fwd, Tm_rev) as a response surface
  slice_primer.png       a 1D slice: R vs one melting temperature, the smooth peak
  convergence.png        best-reward trajectory climbing to the optimum (0.167 -> 1.0)
  convergence.gif        the same climb animated (the demo, as a shareable clip)

The surface is the exact reward math (same weights and membership functions the
verifier uses), plotted over the measurement axes with the other terms held ideal.
The trajectory is a real run of the optimizer, not a drawn curve.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from biorewards.optimize import optimize
from biorewards.mutators import dna_primer_pair_mutator
from biorewards.verifiers.protocol import PrimerPairReward
from biorewards.verifiers.protocol import thermo

INK = "#0B0F17"; PANEL = "#131A26"; GOLD = "#F2B705"; TEXT = "#E6EAF2"
MUT = "#8593ab"; GRID = "#24304a"; RED = "#E1584C"; GREEN = "#35B27A"

plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": INK, "savefig.facecolor": INK,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": GRID, "font.family": "sans-serif", "font.size": 12,
})

OUT = os.path.join(os.path.dirname(__file__), "..", "deck", "figures")
os.makedirs(OUT, exist_ok=True)


# ---- the exact reward math as numpy, matching PrimerPairReward defaults ----

def np_trap(x, a, b, c, d):
    x = np.asarray(x, float); y = np.zeros_like(x)
    up = (x > a) & (x < b); y[up] = (x[up] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    dn = (x > c) & (x < d); y[dn] = (d - x[dn]) / (d - c)
    return np.clip(y, 0, 1)

def np_fall(x, c, d):
    x = np.asarray(x, float); y = np.ones_like(x)
    dn = (x > c) & (x < d); y[dn] = (d - x[dn]) / (d - c)
    y[x >= d] = 0.0
    return np.clip(y, 0, 1)

def reward_surface(tf, tr):
    # weights: tm_fwd 0.2, tm_rev 0.2, tm_match 0.2, and the four ideal-held terms
    # (gc_fwd, gc_rev, gc_clamp, self_complement) contribute 0.4 at score 1.
    return (0.4
            + 0.2 * np_trap(tf, 52, 58, 62, 68)
            + 0.2 * np_trap(tr, 52, 58, 62, 68)
            + 0.2 * np_fall(np.abs(tf - tr), 1, 5))


# ---- figure 1: the response surface (the "function") ----

def fig_landscape():
    g = np.linspace(45, 72, 260)
    TF, TR = np.meshgrid(g, g)
    Z = reward_surface(TF, TR)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "reward", [INK, "#3a3320", "#7a6412", GOLD])
    im = ax.pcolormesh(TF, TR, Z, cmap=cmap, vmin=0, vmax=1, shading="auto")
    ax.contour(TF, TR, Z, levels=[0.6, 0.8, 0.95], colors=[MUT], linewidths=0.6, alpha=0.6)
    ax.axvspan(58, 62, color=GREEN, alpha=0.07)
    ax.axhspan(58, 62, color=GREEN, alpha=0.07)
    ax.plot([60], [60], "o", color=GREEN, ms=9, mec=INK, mew=1.5)
    ax.annotate("optimum\nR = 1.0", (60, 60), (63.5, 49), color=GREEN, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("forward primer  Tm  (C)")
    ax.set_ylabel("reverse primer  Tm  (C)")
    ax.set_title("The reward function is a landscape", color=TEXT, fontsize=15, pad=12, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("reward  R(x)  in [0, 1]", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=MUT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "landscape_primer.png"), dpi=140)
    plt.close(fig)


# ---- figure 2: a 1D slice through the landscape ----

def fig_slice():
    tf = np.linspace(45, 72, 500)
    R = reward_surface(tf, np.full_like(tf, 60.0))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axvspan(58, 62, color=GREEN, alpha=0.10, label="target window 58-62 C")
    ax.plot(tf, R, color=GOLD, lw=2.4)
    ax.set_xlabel("forward primer  Tm  (C)   (reverse held at 60 C)")
    ax.set_ylabel("reward  R(x)")
    ax.set_ylim(0.35, 1.02)
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    ax.set_title("Smooth, bounded, single peak", color=TEXT, fontsize=15, pad=10, loc="left")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slice_primer.png"), dpi=140)
    plt.close(fig)


# ---- run the real optimizer, return the trajectory ----

def run_trajectory(seed, iterations=80, rng_seed=0):
    res = optimize(PrimerPairReward(), seed, dna_primer_pair_mutator,
                   iterations=iterations, population=14, rng_seed=rng_seed, elitism=2)
    best = [float(r["best_reward"]) for r in res.trajectory]
    mean = [float(r["mean_reward"]) for r in res.trajectory]
    cands = [r["best_candidate"] for r in res.trajectory]
    return best, mean, cands, res

SEED = {"fwd": "GCGCACGTACGATCGCGCGCAC", "rev": "AATCAAACAATACAAATAACAC"}
DECEPTIVE = {"fwd": "ATATATATATATATATATAT", "rev": "TATATATATATATATATATA"}


# ---- figure 3: convergence trajectory ----

def fig_convergence(best, mean):
    it = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(it, mean, color=MUT, lw=1.3, alpha=0.8, label="population mean (exploration)")
    ax.plot(it, best, color=GOLD, lw=2.6, label="best so far (monotone)")
    ax.axhline(1.0, color=GREEN, lw=1, ls="--", alpha=0.6)
    ax.plot(0, best[0], "o", color=RED, ms=8, mec=INK, mew=1.4)
    ax.annotate(f"seed  R={best[0]:.3f}", (0, best[0]), (6, best[0] - 0.02), color=RED, fontsize=11)
    ax.plot(it[-1], best[-1], "o", color=GREEN, ms=8, mec=INK, mew=1.4)
    ax.annotate(f"converged  R={best[-1]:.3f}", (it[-1], best[-1]),
                (it[-1] - 34, best[-1] - 0.10), color=GREEN, fontsize=11)
    ax.set_xlabel("iteration")
    ax.set_ylabel("reward  R(x)")
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    ax.set_title("The search converges on the optimum", color=TEXT, fontsize=15, pad=10, loc="left")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=140)
    plt.close(fig)


# ---- figure 4: the animated climb (the shareable clip) ----

def gif_convergence(best, cands):
    it = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, len(best) - 1); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=GREEN, lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("iteration"); ax.set_ylabel("reward  R(x)")
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    ax.set_title("Reward climbing on a primer pair", color=TEXT, fontsize=15, pad=10, loc="left")
    (line,) = ax.plot([], [], color=GOLD, lw=2.6)
    dot, = ax.plot([], [], "o", color=GOLD, ms=8, mec=INK, mew=1.3)
    txt = ax.text(0.03, 0.90, "", transform=ax.transAxes, color=TEXT, fontsize=11,
                  family="monospace", va="top",
                  bbox=dict(boxstyle="round", fc=PANEL, ec=GRID))

    def frame(k):
        line.set_data(it[:k + 1], best[:k + 1])
        dot.set_data([it[k]], [best[k]])
        c = cands[k]
        tf = thermo.tm_nn(c["fwd"]); tr = thermo.tm_nn(c["rev"])
        txt.set_text(f"iter {it[k]:>3}\nR    {best[k]:.3f}\nTm   {tf:4.1f} / {tr:4.1f} C")
        return line, dot, txt

    anim = animation.FuncAnimation(fig, frame, frames=len(best), interval=110, blit=True)
    anim.save(os.path.join(OUT, "convergence.gif"), writer=animation.PillowWriter(fps=9))
    plt.close(fig)


if __name__ == "__main__":
    fig_landscape()
    fig_slice()
    best, mean, cands, res = run_trajectory(SEED)
    fig_convergence(best, mean)
    gif_convergence(best, cands)
    print("seed reward   :", best[0])
    print("final reward  :", best[-1])
    print("iterations    :", len(best))
    print("figures written to deck/figures/")
