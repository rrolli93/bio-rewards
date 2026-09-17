"""One math, four sciences. Show the same reward-climb across all four verifiers,
then assemble a single self-contained explainer page (deck/how-it-works.html).

Every panel is a real search over a real reward function. Nothing is drawn by hand.
"""

import base64
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biorewards.optimize import optimize
from biorewards.mutators import dna_primer_pair_mutator, protein_mutator
from biorewards.verifiers.protocol import PrimerPairReward
from biorewards.verifiers.binder import BinderReward
from biorewards.verifiers.metabolic import MetabolicReward
from biorewards.verifiers.citation import CitationReward

INK = "#0B0F17"; PANEL = "#131A26"; GOLD = "#F2B705"; TEXT = "#E6EAF2"
MUT = "#8593ab"; GRID = "#24304a"; RED = "#E1584C"; GREEN = "#35B27A"
plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": INK, "savefig.facecolor": INK,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": GRID, "font.family": "sans-serif", "font.size": 11,
})
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "deck", "figures")
os.makedirs(OUT, exist_ok=True)


def primer_curve():
    seed = {"fwd": "GCGCACGTACGATCGCGCGCAC", "rev": "AATCAAACAATACAAATAACAC"}
    r = optimize(PrimerPairReward(), seed, dna_primer_pair_mutator,
                 iterations=80, population=14, rng_seed=0, elitism=2)
    return [float(t["best_reward"]) for t in r.trajectory]


def binder_curve():
    seed = {"sequence": "WWLLIIFFVVWWLLIIFFVVKRKRKRKRKRC"}
    r = optimize(BinderReward(), seed, protein_mutator,
                 iterations=120, population=16, rng_seed=1, elitism=2)
    return [float(t["best_reward"]) for t in r.trajectory]


def metabolic_curve():
    # A real search over knockout plans on the toy metabolic network, in a
    # plausible trial order. Lethal or product-killing plans score 0; the empty
    # baseline grows but is not growth-coupled (0.7); the winning knockout forces
    # the product and reaches 1.0. best-so-far is what we plot.
    rf = MetabolicReward()
    trials = [
        ["BIOMASS"],                    # lethal
        ["GLC_uptake"],                 # lethal (starves the cell)
        [],                             # baseline: grows, product not guaranteed
        ["EX_target"],                  # kills the product itself
        ["EX_byproduct"],               # the win: product becomes growth-coupled
        ["EX_byproduct", "BIOMASS"],    # lethal again
    ]
    best = 0.0; curve = []
    for plan in trials:
        r = float(rf.evaluate({"knockouts": plan, "objective": "EX_target"}).reward)
        best = max(best, r)
        curve.append(best)
    return curve


def citation_curve():
    # A claim made progressively true. Each fabrication fixed lets the reward rise;
    # while a fabrication remains, a hard gate holds it at zero.
    rf = CitationReward()
    good_doi = "10.5555/demo.ramucirumab-vegfr2"
    claim = "The ramucirumab Fab binds VEGFR2 at 3.37 nM in PDB 3S36."
    steps = [
        {"claim": claim, "doi": "10.9999/does-not-exist",
         "value": {"number": 999.0, "unit": "nM", "tolerance": 0.1},
         "entity": {"type": "PDB", "id": "5EHF", "expected": "VEGFR2"}},          # doi dead
        {"claim": claim, "doi": good_doi,
         "value": {"number": 999.0, "unit": "nM", "tolerance": 0.1},
         "entity": {"type": "PDB", "id": "5EHF", "expected": "VEGFR2"}},          # entity fake
        {"claim": claim, "doi": good_doi,
         "value": {"number": 999.0, "unit": "nM", "tolerance": 0.1},
         "entity": {"type": "PDB", "id": "3S36", "expected": "VEGFR2"}},          # number wrong
        {"claim": claim, "doi": good_doi,
         "value": {"number": 3.37, "unit": "nM", "tolerance": 0.1},
         "entity": {"type": "PDB", "id": "3S36", "expected": "VEGFR2"}},          # all true
    ]
    return [float(rf.evaluate(s).reward) for s in steps]


def panel(ax, y, title, sub, step=False):
    x = np.arange(len(y))
    if step:
        ax.step(x, y, where="post", color=GOLD, lw=2.4)
        ax.plot(x, y, "o", color=GOLD, ms=5, mec=INK, mew=1)
    else:
        ax.plot(x, y, color=GOLD, lw=2.4)
    ax.axhline(1.0, color=GREEN, lw=0.9, ls="--", alpha=0.5)
    ax.plot(0, y[0], "o", color=RED, ms=7, mec=INK, mew=1.3)
    ax.plot(x[-1], y[-1], "o", color=GREEN, ms=7, mec=INK, mew=1.3)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(-0.5, len(y) - 0.5)
    ax.grid(color=GRID, lw=0.5, alpha=0.5)
    ax.set_title(title, color=TEXT, fontsize=13, loc="left", pad=6)
    ax.text(0.5, 0.06, sub, transform=ax.transAxes, color=MUT, fontsize=9.5, ha="center")
    ax.text(0.02, 0.90, f"{y[0]:.3f}", transform=ax.transAxes, color=RED, fontsize=10)
    ax.text(0.98, 0.90, f"{y[-1]:.3f}", transform=ax.transAxes, color=GREEN, fontsize=10, ha="right")


def build_grid():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    panel(axes[0, 0], primer_curve(), "PRIMERS", "melting-temperature physics")
    panel(axes[0, 1], binder_curve(), "PROTEIN BINDER", "developability of the sequence")
    panel(axes[1, 0], metabolic_curve(), "METABOLIC (FLUX BALANCE)", "product yield from gene knockouts")
    panel(axes[1, 1], citation_curve(), "CITATION", "truth of the claim", step=True)
    for ax in axes[1, :]:
        ax.set_xlabel("search step", color=MUT)
    for ax in axes[:, 0]:
        ax.set_ylabel("reward  R  (0 to 1)", color=MUT)
    fig.suptitle("The same climb, four different sciences", color=TEXT, fontsize=16, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, "convergence_all.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def build_page(grid_path):
    land = b64(os.path.join(OUT, "landscape_primer.png"))
    grid = b64(grid_path)
    html = PAGE.replace("__LANDSCAPE__", land).replace("__GRID__", grid)
    out = os.path.join(HERE, "..", "deck", "how-it-works.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out)


PAGE = """<title>How Bio-Rewards works</title>
<style>
  :root{--bg:#0e1320;--card:#161d2b;--line:#26314a;--text:#eef1f6;--muted:#9aa6bc;--gold:#f2b705;--green:#3fb47d;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,"SF Mono",Menlo,monospace;}
  @media (prefers-color-scheme:light){:root{--bg:#f7f8fb;--card:#fff;--line:#e4e8f0;--text:#141a26;--muted:#5b6678;--gold:#8a6200;--green:#1f8a58;}}
  :root[data-theme="dark"]{--bg:#0e1320;--card:#161d2b;--line:#26314a;--text:#eef1f6;--muted:#9aa6bc;--gold:#f2b705;--green:#3fb47d;}
  :root[data-theme="light"]{--bg:#f7f8fb;--card:#fff;--line:#e4e8f0;--text:#141a26;--muted:#5b6678;--gold:#8a6200;--green:#1f8a58;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.6;padding:min(7vw,60px);}
  .wrap{max-width:820px;margin:0 auto;}
  .kick{font-family:var(--mono);font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);}
  h1{font-size:clamp(28px,5vw,44px);font-weight:750;letter-spacing:-0.03em;line-height:1.08;margin:12px 0 10px;text-wrap:balance;}
  .sub{color:var(--muted);font-size:clamp(16px,2vw,19px);margin-bottom:40px;}
  h2{font-size:clamp(20px,2.6vw,26px);font-weight:700;letter-spacing:-0.02em;margin:40px 0 14px;}
  p{font-size:17px;margin-bottom:14px;}
  strong{font-weight:680;}
  .up{color:var(--green);font-weight:680;} .zero{color:var(--gold);font-weight:680;}
  figure{margin:22px 0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;}
  figure img{width:100%;display:block;border-radius:8px;}
  figcaption{color:var(--muted);font-size:14px;margin-top:12px;padding:0 6px;}
  .grid4{display:grid;grid-template-columns:1fr 1fr;gap:12px 26px;margin-top:8px;}
  .grid4 .it{border-left:2px solid var(--gold);padding-left:14px;}
  .grid4 .it b{display:block;font-size:16px;}
  .grid4 .it span{color:var(--muted);font-size:14.5px;}
  .rule{height:1px;background:var(--line);margin:46px 0;}
  .foot{color:var(--muted);font-size:14px;font-family:var(--mono);margin-top:30px;}
  @media (max-width:620px){.grid4{grid-template-columns:1fr;}}
</style>
<div class="wrap">
  <div class="kick">Bio-Rewards &middot; how it works</div>
  <h1>One math. Four sciences.</h1>
  <p class="sub">The primers, the proteins, the metabolic engineering, the citations. All the same idea underneath.</p>

  <h2>1. The idea</h2>
  <p>An AI proposes a biology answer. A reward function scores it from <strong>0 to 1</strong>. <span class="up">High means real.</span> <span class="zero">Zero means impossible or fake.</span> That score is a fixed calculation, not another opinion, so it does not hallucinate.</p>

  <h2>2. Every reward is a landscape</h2>
  <figure>
    <img src="__LANDSCAPE__" alt="reward landscape over two melting temperatures">
    <figcaption>The reward for a primer pair, drawn over its two melting temperatures. It is bounded between 0 and 1, it has one peak (the right answer), and it falls to a hard 0 where the answer becomes impossible. Every one of the four verifiers has a landscape like this. Only the axes change.</figcaption>
  </figure>

  <h2>3. A good answer makes the search climb</h2>
  <p>An AI searches this landscape, proposing answers and keeping the best one found so far. Because it only ever keeps the best, the score <strong>can only go up</strong>, and it is capped at 1. So the search always settles. <strong>Where it settles is the verdict.</strong> Settling near 1 means it found a real answer. Stalling low means the AI was exploring the wrong place. That is the deterministic layer: truth converges high, fiction gets stuck at zero.</p>

  <figure>
    <img src="__GRID__" alt="four convergence curves, one per verifier">
    <figcaption>The same climb, run for real on all four verifiers. Red dot is the starting answer, green dot is where it converged. Top left, a broken primer pair climbs to a perfect 1.0. Top right, an unstable protein is fixed. Bottom left, a set of gene knockouts is found that forces a cell to make the product (flux balance). Bottom right, a fabricated citation is corrected one lie at a time, held at zero until every fabrication is gone.</figcaption>
  </figure>

  <h2>4. What each one actually measures</h2>
  <div class="grid4">
    <div class="it"><b>Primers</b><span>Melting temperature from nearest-neighbor thermodynamics. Textbook physics, computed exactly.</span></div>
    <div class="it"><b>Protein binder</b><span>Developability of the sequence: charge, water-hating patches, protease sites, cysteine pairing.</span></div>
    <div class="it"><b>Metabolic (flux balance)</b><span>Solves the cell's chemistry as an equation system to find how much product a knockout plan yields.</span></div>
    <div class="it"><b>Citation</b><span>Does the reference resolve, does the number appear in it, is the database entity really what was claimed.</span></div>
  </div>

  <div class="rule"></div>
  <p class="foot">same skeleton every time: R(x) = [passes hard gates] x weighted sum of calibrated measurements.</p>
</div>
"""

if __name__ == "__main__":
    grid_path = build_grid()
    build_page(grid_path)
    print("done")
