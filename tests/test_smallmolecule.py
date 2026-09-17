import pytest

# The small-molecule verifier is the one that leans on rdkit. Skip the whole
# module cleanly when rdkit is absent so the default (no-rdkit) suite stays green.
pytest.importorskip("rdkit")

from decimal import Decimal  # noqa: E402

from biorewards.verifiers.smallmolecule import (  # noqa: E402
    MockDockingProvider,
    SmallMoleculeReward,
)

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
# a 40-carbon alkane: blows MW, cLogP and rotatable-bond count all at once.
GREASE = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"


def test_invalid_smiles_rejected():
    r = SmallMoleculeReward().evaluate({"smiles": "not_a_molecule"})
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("cannot parse" in f for f in r.hard_gate_failures)


def test_unbalanced_parens_rejected():
    r = SmallMoleculeReward().evaluate({"smiles": "C(C(C"})
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")


def test_empty_rejected():
    r = SmallMoleculeReward().evaluate({"smiles": ""})
    assert r.status == "rejected"
    assert any("empty" in f for f in r.hard_gate_failures)


def test_ibuprofen_scores_ok_and_positive():
    r = SmallMoleculeReward().evaluate({"smiles": IBUPROFEN})
    assert r.status == "ok"
    assert r.reward > Decimal("0"), r.as_dict()


def test_ibuprofen_beats_greasy_chain():
    rf = SmallMoleculeReward()
    good = rf.evaluate({"smiles": IBUPROFEN})
    bad = rf.evaluate({"smiles": GREASE})
    assert good.status == "ok"
    assert bad.status == "ok"
    assert good.reward > bad.reward


def test_determinism_byte_identical():
    rf = SmallMoleculeReward()
    a = rf.evaluate({"smiles": IBUPROFEN})
    b = rf.evaluate({"smiles": IBUPROFEN})
    assert a.reward == b.reward
    assert a.evaluation_key == b.evaluation_key
    assert a.as_dict() == b.as_dict()


def test_identity_is_canonical_smiles():
    rf = SmallMoleculeReward()
    # a non-canonical spelling of ibuprofen caches to the same evaluation key.
    a = rf.evaluate({"smiles": IBUPROFEN})
    b = rf.evaluate({"smiles": "OC(=O)C(C)c1ccc(CC(C)C)cc1"})
    assert a.evaluation_key == b.evaluation_key
    assert a.reward == b.reward


def test_mock_docking_provider_deterministic():
    p = MockDockingProvider()
    a = p.dock(IBUPROFEN)
    b = p.dock(IBUPROFEN)
    assert a == b
    assert -11.0 <= a <= -5.0


def test_identity_present_and_in_domain():
    r = SmallMoleculeReward().evaluate({"smiles": IBUPROFEN})
    assert r.applicability.in_domain
    names = {c.name for c in r.components}
    assert {"mw", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "binding"} <= names
