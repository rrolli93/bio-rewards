from decimal import Decimal

import pytest

from biorewards.verifiers.metabolic import MetabolicReward
from biorewards.verifiers.metabolic.providers import (
    KO_BYPRODUCT,
    KO_LETHAL,
    WILD_TYPE,
    MockFBAProvider,
)

OBJ = MockFBAProvider.R_PRODUCT

EMPTY = {"knockouts": list(WILD_TYPE), "objective": OBJ}
GOOD = {"knockouts": list(KO_BYPRODUCT), "objective": OBJ}


def _score(result, name):
    return next(c for c in result.components if c.name == name).score


def test_determinism_byte_identical():
    rf = MetabolicReward()
    a = rf.evaluate(GOOD)
    b = rf.evaluate(GOOD)
    assert a.reward == b.reward
    assert a.evaluation_key == b.evaluation_key


def test_lethal_knockout_rejected():
    r = MetabolicReward().evaluate({"knockouts": list(KO_LETHAL), "objective": OBJ})
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")
    assert any("lethal" in f for f in r.hard_gate_failures)


def test_missing_objective_rejected():
    r = MetabolicReward().evaluate({"knockouts": [], "objective": ""})
    assert r.status == "rejected"
    assert any("objective" in f for f in r.hard_gate_failures)


def test_knockouts_must_be_list():
    r = MetabolicReward().evaluate({"knockouts": "EX_byproduct", "objective": OBJ})
    assert r.status == "rejected"
    assert any("not a list" in f for f in r.hard_gate_failures)


def test_unproducible_objective_rejected():
    r = MetabolicReward().evaluate({"knockouts": [], "objective": "EX_nonexistent"})
    assert r.status == "rejected"
    assert any("not producible" in f for f in r.hard_gate_failures)


def test_byproduct_knockout_beats_empty_design():
    rf = MetabolicReward()
    empty = rf.evaluate(EMPTY)
    good = rf.evaluate(GOOD)
    assert empty.status == "ok"
    assert good.status == "ok"
    assert good.reward > empty.reward, (empty.as_dict(), good.as_dict())


def test_growth_coupling_component_higher_for_coupled_design():
    rf = MetabolicReward()
    empty = rf.evaluate(EMPTY)
    good = rf.evaluate(GOOD)
    # the byproduct knockout forces product at max growth; the empty design does not.
    assert _score(good, "growth_coupling") > _score(empty, "growth_coupling")
    assert _score(empty, "growth_coupling") == Decimal("0")


def test_provider_methods_deterministic():
    p = MockFBAProvider()
    assert p.max_growth([]) == p.max_growth([])
    assert p.max_product([], OBJ) == p.max_product([], OBJ)
    assert p.coupling(KO_BYPRODUCT, OBJ) == p.coupling(KO_BYPRODUCT, OBJ)
    # the toy network's expected optima.
    assert p.max_growth([]) == 10.0
    assert p.coupling([], OBJ) == 0.0          # byproduct absorbs redox, no coupling
    assert p.coupling(KO_BYPRODUCT, OBJ) == 10.0  # now product is forced


def test_lethal_and_viable_growth():
    p = MockFBAProvider()
    assert p.max_growth(KO_LETHAL) == 0.0
    assert p.max_growth(KO_BYPRODUCT) == 10.0


# ---------------------------------------------------------------------------
# Real FBA via cobrapy on the E. coli core textbook model. These run only when
# cobra is installed. The importorskip lives in the fixture (not at module top)
# so the mock tests above still run in the default cobra-free suite.
# ---------------------------------------------------------------------------

ACETATE = "EX_ac_e"
# Cytochrome oxidase. Knocking it out removes the aerobic redox sink and forces
# fermentative acetate, growth-coupling the product.
COMPETING_KO = ["CYTBD"]


@pytest.fixture(scope="module")
def cobra_provider():
    pytest.importorskip("cobra")
    from biorewards.verifiers.metabolic.providers import CobraFBAProvider

    return CobraFBAProvider("textbook")


def test_cobra_wild_type_growth(cobra_provider):
    # Textbook E. coli core wild-type growth is ~0.8739 /h.
    assert cobra_provider.max_growth([]) == pytest.approx(0.8739, abs=1e-3)


def test_cobra_acetate_producible(cobra_provider):
    assert cobra_provider.max_product([], ACETATE) > 1.0


def test_cobra_knockout_changes_product_flux(cobra_provider):
    # At max growth, wild type need not secrete acetate (coupling ~0). Removing
    # the competing redox sink forces acetate, so coupling jumps well above zero.
    wt_coupling = cobra_provider.coupling([], ACETATE)
    ko_coupling = cobra_provider.coupling(COMPETING_KO, ACETATE)
    assert wt_coupling == pytest.approx(0.0, abs=1e-4)
    assert ko_coupling > 1.0
    assert ko_coupling > wt_coupling + 1.0


def test_cobra_provider_methods_deterministic(cobra_provider):
    # FBA is a deterministic LP; results agree to solver tolerance on every call.
    g1, g2 = cobra_provider.max_growth([]), cobra_provider.max_growth([])
    p1, p2 = cobra_provider.max_product([], ACETATE), cobra_provider.max_product([], ACETATE)
    c1, c2 = cobra_provider.coupling(COMPETING_KO, ACETATE), cobra_provider.coupling(COMPETING_KO, ACETATE)
    assert g1 == pytest.approx(g2, abs=1e-6)
    assert p1 == pytest.approx(p2, abs=1e-6)
    assert c1 == pytest.approx(c2, abs=1e-6)


def test_cobra_reward_changes_with_knockout(cobra_provider):
    reward = MetabolicReward(provider=cobra_provider, product_good=20.0, coupling_good=10.0)
    wt = reward.evaluate({"knockouts": [], "objective": ACETATE})
    ko = reward.evaluate({"knockouts": COMPETING_KO, "objective": ACETATE})
    assert wt.status == "ok"
    assert ko.status == "ok"
    assert wt.reward != ko.reward
    # The knockout is the only design that growth-couples acetate.
    def coupling_score(r):
        return next(c for c in r.components if c.name == "growth_coupling").score
    assert coupling_score(ko) > coupling_score(wt)
    assert coupling_score(wt) == Decimal("0")
