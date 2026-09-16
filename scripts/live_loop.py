"""The live loop. Run: uv run python scripts/live_loop.py

The library scores candidates. This closes the loop around it: a probabilistic
generator (a seeded mutator) proposes edits, the deterministic reward function
grades them, and elitist selection keeps what climbs. You watch a bad candidate
get repaired in real time and the reward rise.

Two runs, both offline and deterministic:

  PRIMER  a badly mismatched PCR primer pair (one strand too hot, one too cold)
          pulled into the 58-62C window by PrimerPairReward. The Tm is an exact
          nearest-neighbor calculation, so the convergence you see is real.

  BINDER  an aggregation-prone, Kex2-riddled peptide cleaned up by BinderReward.
          The developability terms (GRAVY, Kex2 sites, hydrophobic run) are exact
          from the sequence; the fold-confidence terms use the mock provider, so
          the meaningful climb here is developability.
"""

from biorewards.mutators import dna_primer_pair_mutator, protein_mutator
from biorewards.optimize import optimize
from biorewards.verifiers.binder import BinderReward
from biorewards.verifiers.binder import sequence as protein
from biorewards.verifiers.protocol import PrimerPairReward, thermo

RULE = "=" * 72
THIN = "-" * 72


def _primer_metrics(pair):
    return {
        "tm_f": thermo.tm_nn(pair["fwd"]),
        "tm_r": thermo.tm_nn(pair["rev"]),
        "gc_f": thermo.gc_content(pair["fwd"]),
        "gc_r": thermo.gc_content(pair["rev"]),
    }


def run_primer():
    print(f"\n{RULE}\nPRIMER RUN  (PrimerPairReward + dna_primer_pair_mutator)\n{RULE}")
    print("Goal: pull both strands into the 58-62C window and match them.")
    print("Tm is exact SantaLucia-98 nearest-neighbor, not a heuristic.\n")

    rf = PrimerPairReward()
    # fwd too hot and GC-rich, rev too cold and AT-rich: a ~20C mismatch.
    seed = {"fwd": "GCGCACGTACGATCGCGCGCAC", "rev": "AATCAAACAATACAAATAACAC"}
    seed_reward = rf.evaluate(seed).reward
    m0 = _primer_metrics(seed)

    result = optimize(rf, seed, dna_primer_pair_mutator, iterations=200, rng_seed=0)

    header = f"{'iter':>5}  {'reward':>9}  {'fwd Tm':>7}  {'rev Tm':>7}  {'fwd GC':>7}  {'rev GC':>7}"
    print(header)
    print(THIN)
    for rec in result.trajectory:
        if rec["iteration"] % 20 == 0:
            m = _primer_metrics(rec["best_candidate"])
            print(f"{rec['iteration']:>5}  {rec['best_reward']!s:>9}  "
                  f"{m['tm_f']:>6.1f}C  {m['tm_r']:>6.1f}C  "
                  f"{m['gc_f']:>6.0f}%  {m['gc_r']:>6.0f}%")

    best = result.best_candidate
    mb = _primer_metrics(best)
    print(THIN)
    print("BEFORE -> AFTER")
    print(f"  reward   {seed_reward}  ->  {result.best_reward}")
    print(f"  fwd Tm   {m0['tm_f']:.1f}C  ->  {mb['tm_f']:.1f}C   (window 58-62C)")
    print(f"  rev Tm   {m0['tm_r']:.1f}C  ->  {mb['tm_r']:.1f}C   (window 58-62C)")
    print(f"  dTm      {abs(m0['tm_f'] - m0['tm_r']):.1f}C  ->  {abs(mb['tm_f'] - mb['tm_r']):.1f}C")
    print(f"  fwd      {seed['fwd']}  ->  {best['fwd']}")
    print(f"  rev      {seed['rev']}  ->  {best['rev']}")


def _binder_metrics(seq):
    return {
        "gravy": protein.gravy(seq),
        "kex2": protein.kex2_sites(seq),
        "patch": protein.longest_hydrophobic_run(seq),
        "cys": protein.cysteine_count(seq),
        "len": len(seq),
    }


def run_binder():
    print(f"\n{RULE}\nBINDER RUN  (BinderReward + protein_mutator)\n{RULE}")
    print("Goal: clean up an aggregation-prone, Kex2-riddled peptide.")
    print("Developability is exact from the sequence. Fold terms use the mock")
    print("provider, so the meaningful climb here is developability.\n")

    rf = BinderReward()
    # long hydrophobic runs (aggregation), KR/RR dibasics (Kex2 clipping), odd Cys.
    seed = {"sequence": "WWLLIIFFVVWWLLIIFFVVKRKRKRKRKRC"}
    seed_reward = rf.evaluate(seed).reward
    m0 = _binder_metrics(seed["sequence"])

    result = optimize(rf, seed, protein_mutator, iterations=200, rng_seed=0)

    header = f"{'iter':>5}  {'reward':>9}  {'GRAVY':>7}  {'Kex2':>5}  {'hphob run':>9}  {'Cys':>4}"
    print(header)
    print(THIN)
    for rec in result.trajectory:
        if rec["iteration"] % 20 == 0:
            m = _binder_metrics(rec["best_candidate"]["sequence"])
            print(f"{rec['iteration']:>5}  {rec['best_reward']!s:>9}  "
                  f"{m['gravy']:>7.2f}  {m['kex2']:>5}  {m['patch']:>9}  {m['cys']:>4}")

    best_seq = result.best_candidate["sequence"]
    mb = _binder_metrics(best_seq)
    print(THIN)
    print("BEFORE -> AFTER")
    print(f"  reward       {seed_reward}  ->  {result.best_reward}")
    print(f"  GRAVY        {m0['gravy']:+.2f}  ->  {mb['gravy']:+.2f}   (soluble sweet spot near -0.3)")
    print(f"  Kex2 sites   {m0['kex2']}  ->  {mb['kex2']}   (KR/RR dibasics, want 0)")
    print(f"  hphob run    {m0['patch']}  ->  {mb['patch']}   (longest hydrophobic run, want short)")
    print(f"  Cys count    {m0['cys']} ({'even' if m0['cys'] % 2 == 0 else 'ODD'})  ->  "
          f"{mb['cys']} ({'even' if mb['cys'] % 2 == 0 else 'ODD'})")
    print(f"  sequence     {seed['sequence']}")
    print(f"            -> {best_seq}")


if __name__ == "__main__":
    run_primer()
    run_binder()
    print()
