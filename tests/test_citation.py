from decimal import Decimal

import pytest

from biorewards.verifiers.citation import CitationReward
from biorewards.verifiers.citation.providers import MockResolver

RAMUCIRUMAB_DOI = "10.5555/demo.ramucirumab-vegfr2"

# The fabrication from the real anti-drift incident: 5EHF (a laccase) claimed as
# a VEGFR2 complex, with a number that is nowhere in the source.
FABRICATION = {
    "claim": "PDB 5EHF is the crystal structure of the VEGFR2 complex "
    "bound with 3.37 nM affinity.",
    "doi": RAMUCIRUMAB_DOI,
    "value": {"number": 3.37, "unit": "nM", "tolerance": 0.1},
    "entity": {"type": "PDB", "id": "5EHF", "expected": "VEGFR2"},
}

# The correct claim: 3S36 really is the VEGFR2 / ramucirumab co-crystal, and the
# 3.37 nM figure is in the resolved abstract.
CORRECT = {
    "claim": "The ramucirumab Fab binds VEGFR2 with a dissociation constant of "
    "3.37 nM, as shown in the co-crystal structure PDB 3S36.",
    "doi": RAMUCIRUMAB_DOI,
    "value": {"number": 3.37, "unit": "nM", "tolerance": 0.1},
    "entity": {"type": "PDB", "id": "3S36", "expected": "VEGFR2"},
}


def _component(result, name):
    return next(c for c in result.components if c.name == name)


def test_determinism_byte_identical():
    rf = CitationReward()
    a = rf.evaluate(CORRECT)
    b = rf.evaluate(CORRECT)
    assert a.reward == b.reward
    assert a.evaluation_key == b.evaluation_key


def test_unresolvable_doi_rejected():
    r = CitationReward().evaluate(
        {"claim": "some claim", "doi": "10.9999/does-not-exist"}
    )
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("resolve" in f for f in r.hard_gate_failures)


def test_invalid_doi_syntax_rejected():
    r = CitationReward().evaluate({"claim": "some claim", "doi": "not-a-doi"})
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("valid DOI" in f for f in r.hard_gate_failures)


def test_missing_doi_rejected():
    r = CitationReward().evaluate({"claim": "some claim", "doi": ""})
    assert r.status == "rejected"
    assert any("missing" in f for f in r.hard_gate_failures)


def test_fabrication_hard_gated_on_entity():
    r = CitationReward().evaluate(FABRICATION)
    # 5EHF is a laccase, not VEGFR2. A known-and-contradicted provenance is a
    # fabrication, so the whole claim hard-gates to zero, not a soft deduction.
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("provenance contradicted" in f for f in r.hard_gate_failures)


def test_correct_claim_scores_high():
    r = CitationReward().evaluate(CORRECT)
    assert r.status == "ok"
    assert _component(r, "entity_type_correct").score == Decimal("1")
    assert _component(r, "number_in_source").score == Decimal("1")
    assert r.reward > Decimal("0.7"), r.as_dict()


def test_fabrication_scores_lower_than_correct():
    fab = CitationReward().evaluate(FABRICATION).reward
    good = CitationReward().evaluate(CORRECT).reward
    assert fab < good


def test_number_absent_scores_zero():
    x = {
        "claim": "The ramucirumab Fab binds VEGFR2 with 999.0 nM affinity.",
        "doi": RAMUCIRUMAB_DOI,
        "value": {"number": 999.0, "unit": "nM", "tolerance": 0.1},
    }
    r = CitationReward().evaluate(x)
    assert r.status == "ok"
    assert _component(r, "number_in_source").score == Decimal("0")


def test_components_omitted_when_not_asserted():
    # No value and no entity: only claim_support should be present.
    x = {"claim": "ramucirumab targets VEGFR2", "doi": RAMUCIRUMAB_DOI}
    r = CitationReward().evaluate(x)
    names = {c.name for c in r.components}
    assert names == {"claim_support"}


def test_mock_resolver_deterministic():
    res = MockResolver()
    assert res.resolve_doi(RAMUCIRUMAB_DOI) == res.resolve_doi(RAMUCIRUMAB_DOI)
    assert res.entity_identity("PDB", "5EHF") == "laccase"
    assert res.entity_identity("PDB", "5ehf") == "laccase"
    assert res.resolve_doi("10.9999/does-not-exist") is None
    assert res.entity_identity("PDB", "0XXX") is None


# ---------------------------------------------------------------------------
# Live resolution against Crossref + RCSB. These need both the requests package
# and network. The importorskip lives in the fixture (not at module top) so the
# mock tests above still run in the default suite. Any connection failure skips,
# so the default offline suite never fails on these.
# ---------------------------------------------------------------------------

# A DOI that resolves in Crossref with a real (JATS-tagged) abstract.
LIVE_DOI = "10.1038/s41586-020-2649-2"


@pytest.fixture
def live_resolver():
    pytest.importorskip("requests")
    from biorewards.verifiers.citation.providers import LiveResolver

    return LiveResolver()


def _live_or_skip(fn):
    """Run fn(); skip the test on any network/connection error rather than fail."""
    import requests

    try:
        return fn()
    except requests.RequestException as exc:  # pragma: no cover (network-dependent)
        pytest.skip(f"network unavailable: {exc}")


def test_live_doi_resolves(live_resolver):
    meta = _live_or_skip(lambda: live_resolver.resolve_doi(LIVE_DOI))
    if meta is None:
        pytest.skip("Crossref returned no record (network or rate limit)")
    assert "numpy" in meta["title"].lower()
    assert meta["year"] == 2020
    # abstract is present and the JATS tags were stripped.
    assert meta["abstract"]
    assert "<jats" not in meta["abstract"]


def test_live_pdb_entity_identity(live_resolver):
    ident = _live_or_skip(lambda: live_resolver.entity_identity("PDB", "3S36"))
    if ident is None:
        pytest.skip("RCSB returned no record (network or rate limit)")
    # 3S36 is the anti-VEGF-receptor (VEGFR2) antibody structure.
    assert "vegf" in ident.lower()


def test_live_5ehf_is_laccase(live_resolver):
    # Confirms the mock's 5EHF=laccase trap against the live database.
    ident = _live_or_skip(lambda: live_resolver.entity_identity("PDB", "5EHF"))
    if ident is None:
        pytest.skip("RCSB returned no record (network or rate limit)")
    assert "laccase" in ident.lower()


def test_live_reward_end_to_end(live_resolver):
    claim = {
        "claim": "NumPy is the primary array programming library for the Python "
        "language, operating on vectors and matrices.",
        "doi": LIVE_DOI,
    }
    result = _live_or_skip(
        lambda: CitationReward(resolver=live_resolver).evaluate(claim)
    )
    if result.status == "rejected":
        pytest.skip("live DOI did not resolve (network or rate limit)")
    assert result.status == "ok"
    assert {c.name for c in result.components} == {"claim_support"}
