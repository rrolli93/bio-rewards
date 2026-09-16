from decimal import Decimal

from biorewards.verifiers.binder import BinderReward, MockFoldProvider

# A clean, well-behaved designed binder: mixed composition, mild charge, one
# disulfide pair (even Cys), no KR/RR dibasic motifs, no long hydrophobic run.
GOOD = {"sequence": "SEDKEAWNTGCKQFVDSNGRTHCLDPEKAQY", "target": "OX2R"}


def test_determinism_byte_identical():
    rf = BinderReward()
    a = rf.evaluate(GOOD)
    b = rf.evaluate(GOOD)
    assert a.reward == b.reward
    assert a.evaluation_key == b.evaluation_key


def test_non_standard_residue_rejected():
    r = BinderReward().evaluate({"sequence": "SEDKEAWNTGCKQFVDSXGRTHCLDPEKAQY"})
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("non-standard" in f for f in r.hard_gate_failures)


def test_length_gate_too_short():
    r = BinderReward().evaluate({"sequence": "ACDEFG"})
    assert r.status == "rejected"
    assert any("length" in f for f in r.hard_gate_failures)


def test_length_gate_too_long():
    r = BinderReward().evaluate({"sequence": "A" * 200})
    assert r.status == "rejected"
    assert any("length" in f for f in r.hard_gate_failures)


def test_empty_rejected():
    r = BinderReward().evaluate({"sequence": ""})
    assert r.status == "rejected"
    assert any("empty" in f for f in r.hard_gate_failures)


def test_good_binder_scores_ok_and_positive():
    r = BinderReward().evaluate(GOOD)
    assert r.status == "ok"
    assert r.reward > Decimal("0"), r.as_dict()


def _component_score(result, name):
    return next(c.score for c in result.components if c.name == name)


def test_odd_cysteine_scores_lower_than_even():
    even = {"sequence": "SEDKEAWNTGCKQFVDSNGRTHCLDPEKAQY"}   # 2 Cys
    odd = {"sequence": "SEDKEAWNTGCKQFVDSNGRTHLDPEKAQY"}     # 1 Cys
    even_r = BinderReward().evaluate(even)
    odd_r = BinderReward().evaluate(odd)
    assert _component_score(odd_r, "cysteine_parity") < _component_score(even_r, "cysteine_parity")


def test_kex2_repeats_score_lower_than_none():
    kr_heavy = {"sequence": "SKRDKRAEKRNTKRQFKRVDKRSGKRTHKR"}  # many KR motifs
    clean = {"sequence": "SEDAEANTGAKQFVDSNGATHALDPEKAQY"}       # no KR/RR
    heavy_r = BinderReward().evaluate(kr_heavy)
    clean_r = BinderReward().evaluate(clean)
    assert _component_score(heavy_r, "kex2_sites") < _component_score(clean_r, "kex2_sites")


def test_mock_fold_provider_deterministic():
    p = MockFoldProvider()
    seq = GOOD["sequence"]
    a = p.fold(seq)
    b = p.fold(seq)
    assert a == b
    assert set(a) == {"plddt", "pae_interaction", "self_consistency_rmsd"}
    assert 40.0 <= a["plddt"] <= 95.0
    assert 3.0 <= a["pae_interaction"] <= 28.0
    assert 0.5 <= a["self_consistency_rmsd"] <= 6.0
