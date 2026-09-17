from decimal import Decimal

from biorewards.verifiers.binder import BinderReward, MockFoldProvider
from biorewards.verifiers.binder.ipsae import compute_interface_metrics

# A clean, well-behaved designed binder: mixed composition, mild charge, one
# disulfide pair (even Cys), no KR/RR dibasic motifs, no long hydrophobic run.
GOOD = {"sequence": "SEDKEAWNTGCKQFVDSNGRTHCLDPEKAQY", "target": "MNLEACGRTWQ"}


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
    tgt = GOOD["target"]
    a = p.fold(seq, tgt)
    b = p.fold(seq, tgt)
    assert a == b
    assert set(a) == {"plddt", "iptm", "pae", "n_binder", "n_target"}
    assert 50.0 <= a["plddt"] <= 95.0
    assert 0.2 <= a["iptm"] <= 0.9
    assert a["n_binder"] == len(seq)
    assert a["n_target"] == len(tgt)
    assert len(a["pae"]) == len(seq) + len(tgt)


def test_mock_fold_provider_monomer_has_no_iptm():
    p = MockFoldProvider()
    a = p.fold(GOOD["sequence"], None)
    assert a["iptm"] is None
    assert a["n_target"] == 0
    assert len(a["pae"]) == len(GOOD["sequence"])


def test_ipsae_tiny_cross_pae_scores_high():
    # 4x4 (2 binder + 2 target) PAE, everything a near-zero 0.1 A. ipSAE tends to
    # 1.0 only when PAE sits well under the d0 radius (which clamps at 1.0 A).
    pae = [[0.1] * 4 for _ in range(4)]
    m = compute_interface_metrics(pae, 2, 2)
    assert m["ipsae"] > 0.95
    assert m["pae_interaction"] < 2.0


def test_ipsae_huge_cross_pae_scores_low():
    # cross-chain PAE at 29 A is above the 10 A cutoff => no confident pairs.
    pae = [[29.0] * 4 for _ in range(4)]
    m = compute_interface_metrics(pae, 2, 2)
    assert m["ipsae"] < 0.05
    assert m["pae_interaction"] > 25.0


def test_interface_components_present_with_target():
    r = BinderReward().evaluate(GOOD)
    assert r.status == "ok"
    names = {c.name for c in r.components}
    assert {"iptm", "ipsae", "pae_interaction"} <= names
    assert "plddt" in names


def test_interface_components_absent_without_target():
    r = BinderReward().evaluate({"sequence": GOOD["sequence"]})
    assert r.status == "ok"
    names = {c.name for c in r.components}
    assert not ({"iptm", "ipsae", "pae_interaction"} & names)
    assert "plddt" in names


def test_complex_determinism_byte_identical():
    rf = BinderReward()
    a = rf.evaluate(GOOD)
    b = rf.evaluate(GOOD)
    assert a.reward == b.reward
    assert a.evaluation_key == b.evaluation_key
    assert a.as_dict() == b.as_dict()


def _ipsae_raw(result):
    return next(c.raw for c in result.components if c.name == "ipsae")


def test_good_complex_scores_higher_than_bad_complex():
    # Same clean binder sequence; MockFoldProvider derives interface quality from
    # sha256(binder/target). "PEPTAI" hashes to a high-q complex (low interface
    # PAE, high ipTM, high ipSAE, high pLDDT); "ABCDEF" to a low-q one.
    rf = BinderReward()
    seq = GOOD["sequence"]
    good = rf.evaluate({"sequence": seq, "target": "PEPTAI"})
    bad = rf.evaluate({"sequence": seq, "target": "ABCDEF"})
    assert good.reward > bad.reward
    assert _ipsae_raw(good) > _ipsae_raw(bad)
