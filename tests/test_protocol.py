from decimal import Decimal

from biorewards.verifiers.protocol import GuideReward, PrimerPairReward
from biorewards.verifiers.protocol import thermo

# Classic well-behaved GAPDH primer pair (positive control).
GOOD = {"fwd": "ACCACAGTCCATGCCATCAC", "rev": "TCCACCACCCTGTTGCTGTA"}


def test_tm_in_expected_band():
    tm = thermo.tm_nn(GOOD["fwd"])
    assert 52 < tm < 66, f"GAPDH fwd Tm {tm:.1f}C outside expected band"


def test_good_pair_scores_high():
    r = PrimerPairReward().evaluate(GOOD)
    assert r.status == "ok"
    assert r.reward > Decimal("0.7"), r.as_dict()


def test_determinism_byte_identical():
    rf = PrimerPairReward()
    a = rf.evaluate(GOOD)
    b = rf.evaluate(GOOD)
    assert a.reward == b.reward
    assert a.evaluation_key == b.evaluation_key


def test_non_acgt_rejected():
    r = PrimerPairReward().evaluate({"fwd": "ACCACAGTCCAUGCCATCAC", "rev": GOOD["rev"]})
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("non-ACGT" in f for f in r.hard_gate_failures)


def test_length_gate():
    r = PrimerPairReward().evaluate({"fwd": "ACGT", "rev": GOOD["rev"]})
    assert r.status == "rejected"
    assert any("length" in f for f in r.hard_gate_failures)


def test_gc_hairpin_prone_scores_lower_than_good():
    bad = {"fwd": "GCGCGCGCGCGCGCGCGCGC", "rev": "GCGCGCGCGCGCGCGCGCGC"}
    good_r = PrimerPairReward().evaluate(GOOD).reward
    bad_r = PrimerPairReward().evaluate(bad).reward
    assert bad_r < good_r


# ---- CRISPR guide ----

def test_guide_without_pam_rejected():
    r = GuideReward().evaluate({"guide": "GACGCATCGTACGATCGTAC", "pam": "TAA"})
    assert r.status == "rejected"
    assert any("NGG" in f for f in r.hard_gate_failures)


def test_guide_polyT_terminator_rejected():
    r = GuideReward().evaluate({"guide": "GACGCTTTTACGATCGTACG", "pam": "AGG"})
    assert r.status == "rejected"
    assert any("poly-T" in f for f in r.hard_gate_failures)


def test_good_guide_ok():
    r = GuideReward().evaluate({"guide": "GACGCATCGTACGATCGTAC", "pam": "AGG"})
    assert r.status == "ok"
    assert r.reward > Decimal("0")
