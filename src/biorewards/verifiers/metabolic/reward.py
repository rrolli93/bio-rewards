"""Reward function for metabolic-engineering knockout strategies.

An agent proposes a set of gene/reaction knockouts and a product objective. The
deterministic ground truth is flux balance analysis: a linear program with one
optimum. A design is good when it makes a lot of product while staying viable,
and best when it is growth-coupled (product is forced whenever the cell grows).

The reward is pure and offline through MockFBAProvider. CobraFBAProvider swaps
in a genome-scale model without changing any scoring logic.
"""

from __future__ import annotations

import json
from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import anchored, rising

from .providers import EPS, FBAProvider, MockFBAProvider


class MetabolicReward(RewardFunction):
    """Score x = {"knockouts": [...], "objective": "<product reaction id>"}.

    The anchors default to the MockFBAProvider toy network (theoretical max
    product flux 10). For a real model set product_good / coupling_good from a
    reference panel of known strain designs.
    """

    name = "metabolic_knockout"
    version = "0.1.0"

    def __init__(
        self,
        provider: FBAProvider | None = None,
        min_growth_frac: float = 0.1,
        product_good: float = 10.0,
        coupling_good: float = 10.0,
        growth_retained_edges=(0.1, 0.5),
    ):
        self.provider = provider if provider is not None else MockFBAProvider()
        self.min_growth_frac = min_growth_frac
        self.product_good = product_good
        self.coupling_good = coupling_good
        self.growth_retained_edges = growth_retained_edges

    def config(self) -> dict:
        return {
            "provider": type(self.provider).__name__,
            "min_growth_frac": self.min_growth_frac,
            "product_good": self.product_good,
            "coupling_good": self.coupling_good,
            "growth_retained_edges": self.growth_retained_edges,
        }

    def identity(self, x) -> str:
        kos = sorted(str(k) for k in x.get("knockouts", []) or [])
        obj = str(x.get("objective", ""))
        return json.dumps(
            {"knockouts": kos, "objective": obj},
            sort_keys=True,
            separators=(",", ":"),
        )

    def hard_gates(self, x) -> list[str]:
        fails = []
        obj = x.get("objective", "")
        if not obj:
            fails.append("objective: missing or empty")
        kos = x.get("knockouts", [])
        if not isinstance(kos, list):
            fails.append("knockouts: not a list")
        # provider-touching gates only once the inputs are structurally sane.
        if fails:
            return fails
        if self.provider.max_growth(kos) <= EPS:
            fails.append("lethal: no viable growth")
        # producibility is a property of the model at wild type, not of the design.
        if self.provider.max_product([], obj, min_growth_frac=0.0) <= EPS:
            fails.append(f"objective '{obj}': not producible in the model")
        return fails

    def score_components(self, x) -> list[Component]:
        kos = x["knockouts"]
        obj = x["objective"]
        wt_growth = self.provider.max_growth([])

        max_prod = self.provider.max_product(kos, obj, self.min_growth_frac)
        max_grow = self.provider.max_growth(kos)
        coupled = self.provider.coupling(kos, obj)
        grow_frac = max_grow / wt_growth if wt_growth > EPS else 0.0
        a, b = self.growth_retained_edges

        return [
            Component(
                "product_yield", round(max_prod, 4),
                anchored(max_prod, good=self.product_good, bad=0),
                Decimal("0.5"), f"max product flux {max_prod:.3f}",
            ),
            Component(
                "growth_retained", round(grow_frac, 4),
                rising(grow_frac, a, b),
                Decimal("0.2"), f"{grow_frac * 100:.0f}% of wild-type growth",
            ),
            Component(
                "growth_coupling", round(coupled, 4),
                anchored(coupled, good=self.coupling_good, bad=0),
                Decimal("0.3"),
                f"guaranteed product flux at max growth {coupled:.3f}",
            ),
        ]

    def applicability(self, x) -> Applicability:
        toy = isinstance(self.provider, MockFBAProvider)
        return Applicability(
            in_domain=True,
            reason=(
                "FBA gives a deterministic LP optimum but assumes metabolic "
                "steady state, ignores enzyme kinetics and regulation, and is "
                "only as good as the stoichiometric model. "
                + (
                    "MockFBAProvider is a toy network for demonstration; "
                    "CobraFBAProvider on a genome-scale model is the real use."
                    if toy
                    else "Genome-scale model in use via cobrapy."
                )
            ),
        )
