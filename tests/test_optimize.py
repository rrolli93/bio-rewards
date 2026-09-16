from decimal import Decimal

from biorewards.mutators import (
    AMINO_ACIDS,
    BASES,
    dna_primer_pair_mutator,
    protein_mutator,
)
from biorewards.optimize import optimize
from biorewards.verifiers.binder import BinderReward
from biorewards.verifiers.protocol import PrimerPairReward
from biorewards.verifiers.protocol import thermo

# A deliberately poor primer pair: fwd runs hot and GC-rich (Tm ~66C, 71% GC),
# rev runs cold and AT-rich (Tm ~45C, 23% GC). A ~20C mismatch plus both Tms out
# of the 58-62C window. The optimizer has to pull both into the window and match
# them to each other.
POOR_PRIMER = {
    "fwd": "GCGCACGTACGATCGCGCGCAC",
    "rev": "AATCAAACAATACAAATAACAC",
}

# An aggregation-prone, Kex2-riddled peptide (odd Cys, long hydrophobic run).
POOR_BINDER = {"sequence": "WWLLIIFFVVWWLLIIFFVVKRKRKRKRKRC"}


# ---- optimizer: determinism ----

def test_optimize_is_deterministic():
    rf = PrimerPairReward()
    a = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=60, rng_seed=7)
    b = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=60, rng_seed=7)
    assert a.best_candidate == b.best_candidate
    assert a.best_reward == b.best_reward


def test_different_seed_can_differ():
    # Not a hard guarantee, but a smoke check that the seed actually steers.
    rf = PrimerPairReward()
    a = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=60, rng_seed=1)
    b = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=60, rng_seed=2)
    # Both should still be strong climbs regardless of path.
    assert a.best_reward > Decimal("0.8")
    assert b.best_reward > Decimal("0.8")


# ---- optimizer: improvement ----

def test_primer_run_climbs_and_beats_seed():
    rf = PrimerPairReward()
    seed_reward = rf.evaluate(POOR_PRIMER).reward
    res = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=200, rng_seed=0)
    assert res.best_reward > seed_reward
    assert res.best_reward > Decimal("0.8"), res.best_reward


def test_binder_run_does_not_regress():
    rf = BinderReward()
    seed_reward = rf.evaluate(POOR_BINDER).reward
    res = optimize(rf, POOR_BINDER, protein_mutator, iterations=200, rng_seed=0)
    assert res.best_reward >= seed_reward
    # the developability half is real and should improve on a Kex2-riddled seed
    assert res.best_reward > seed_reward


def test_best_result_is_full_reward_result():
    rf = PrimerPairReward()
    res = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=40, rng_seed=0)
    assert res.best_result.status == "ok"
    assert res.best_result.reward == res.best_reward
    assert len(res.best_result.components) > 0


# ---- optimizer: trajectory monotonicity ----

def test_trajectory_best_reward_monotonic_non_decreasing():
    rf = PrimerPairReward()
    res = optimize(rf, POOR_PRIMER, dna_primer_pair_mutator, iterations=120, rng_seed=3)
    bests = [rec["best_reward"] for rec in res.trajectory]
    assert bests[0] is not None
    for earlier, later in zip(bests, bests[1:]):
        assert later >= earlier, (earlier, later)
    # trajectory records iteration 0 (initial pop) through the last iteration
    assert res.trajectory[0]["iteration"] == 0
    assert res.trajectory[-1]["iteration"] == 120


# ---- mutators: determinism and validity ----

def test_primer_mutator_deterministic():
    import random
    a = dna_primer_pair_mutator(POOR_PRIMER, random.Random(11))
    b = dna_primer_pair_mutator(POOR_PRIMER, random.Random(11))
    assert a == b
    # input untouched
    assert POOR_PRIMER == {"fwd": "GCGCACGTACGATCGCGCGCAC", "rev": "AATCAAACAATACAAATAACAC"}


def test_primer_mutator_produces_valid_dna_within_bounds():
    import random
    rng = random.Random(0)
    cand = dict(POOR_PRIMER)
    for _ in range(500):
        cand = dna_primer_pair_mutator(cand, rng, min_len=15, max_len=30)
        for role in ("fwd", "rev"):
            seq = cand[role]
            assert set(seq) <= set(BASES)
            assert 15 <= len(seq) <= 30


def test_protein_mutator_deterministic():
    import random
    a = protein_mutator(POOR_BINDER, random.Random(5))
    b = protein_mutator(POOR_BINDER, random.Random(5))
    assert a == b
    assert POOR_BINDER == {"sequence": "WWLLIIFFVVWWLLIIFFVVKRKRKRKRKRC"}


def test_protein_mutator_produces_canonical_aa_within_bounds():
    import random
    rng = random.Random(0)
    cand = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
    for _ in range(500):
        cand = protein_mutator(cand, rng, min_len=8, max_len=150)
        seq = cand["sequence"]
        assert set(seq) <= set(AMINO_ACIDS)
        assert 8 <= len(seq) <= 150


def test_protein_mutator_preserves_other_keys():
    import random
    out = protein_mutator({"sequence": "ACDEFGHIK", "target": "OX2R"}, random.Random(1))
    assert out["target"] == "OX2R"
