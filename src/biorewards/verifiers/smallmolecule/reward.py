"""Reward function for small-molecule (drug-like) candidates.

SmallMoleculeReward -- score a candidate small molecule the way a med chemist
would before spending on it, the same shape the molecule-optimization-sandbox
used: deterministic drug-likeness, plus synthetic accessibility, plus a binding
term from an injected docking provider.

  * drug-likeness, computed exactly from the structure (MW, cLogP, TPSA, H-bond
    donors/acceptors, rotatable bonds) with the sandbox's trapezoid sweet spots
  * synthetic accessibility, the Ertl-Schuffenhauer SA_Score (lower is easier)
  * binding, a predicted docking affinity from an injected DockingProvider

The drug-likeness and SA halves are deterministic and offline. The binding half
is only as real as the provider: with MockDockingProvider it is illustrative,
with Rowan / Vina it is a predicted affinity. Applicability says which is which.
A PAINS structural alert is a hard reject, not a soft penalty, mirroring the
sandbox's reactive-group gate.
"""

from __future__ import annotations

from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import anchored, falling, trapezoid

from . import chem
from .providers import DockingProvider, MockDockingProvider


class SmallMoleculeReward(RewardFunction):
    """Drug-like small molecule. Expects x = {'smiles': '<SMILES>'}.

    identity(x) is the RDKit canonical SMILES (or the raw string if it will not
    parse), so the score caches by structure rather than by input spelling.
    """

    name = "small_molecule"
    version = "0.1.0"

    def __init__(self, docking_provider: DockingProvider | None = None):
        self.docking_provider = docking_provider or MockDockingProvider()

    def config(self) -> dict:
        return {
            "docking_provider": getattr(
                self.docking_provider, "name", type(self.docking_provider).__name__
            ),
        }

    def identity(self, x) -> str:
        return chem.canonical(x.get("smiles", ""))

    def hard_gates(self, x) -> list[str]:
        smiles = x.get("smiles", "")
        if not smiles:
            return ["smiles: empty"]
        mol = chem.parse(smiles)
        if mol is None:
            return ["smiles: RDKit cannot parse"]
        # a PAINS hit is a reactive / pan-assay-interference structural alert:
        # a hard reject, the same way the sandbox gated reactive groups.
        return [f"PAINS: {name}" for name in chem.pains_hits(mol)]

    def score_components(self, x) -> list[Component]:
        mol = chem.parse(x["smiles"])
        d = chem.descriptors(mol)
        mw = d["mw"]
        clogp = d["clogp"]
        tpsa = d["tpsa"]
        hbd = d["hbd"]
        hba = d["hba"]
        rot = d["rotatable_bonds"]

        components = [
            # drug-likeness sweet spots, lifted straight from the sandbox anchors
            Component("mw", round(mw, 2), trapezoid(mw, 80, 200, 450, 650), Decimal("0.15"),
                      f"{mw:.0f} Da"),
            Component("clogp", round(clogp, 2), trapezoid(clogp, -3, 0.5, 4.0, 6.5),
                      Decimal("0.15"), f"{clogp:.2f}"),
            Component("tpsa", round(tpsa, 2), trapezoid(tpsa, 0, 20, 120, 180), Decimal("0.12"),
                      f"{tpsa:.0f} A^2"),
            Component("hbd", hbd, falling(hbd, 3, 6), Decimal("0.06"), f"{hbd} donors"),
            Component("hba", hba, falling(hba, 8, 13), Decimal("0.06"), f"{hba} acceptors"),
            Component("rotatable_bonds", rot, falling(rot, 6, 14), Decimal("0.06"),
                      f"{rot} rotatable"),
        ]

        # synthetic accessibility: lower is easier to make. Omitted (not scored 0)
        # when the RDKit Contrib SA_Score module is unavailable at runtime.
        sa = chem.sa_score(mol)
        if sa is not None:
            components.append(
                Component("synthetic_access", round(sa, 2), anchored(sa, 2.0, 8.0),
                          Decimal("0.10"), f"SA {sa:.1f}"))

        # binding: predicted docking affinity, more negative is better. Dock the
        # canonical SMILES so the score is a function of structure, not spelling.
        affinity = self.docking_provider.dock(chem.canonical(x["smiles"]))
        components.append(
            Component("binding", round(affinity, 2), anchored(affinity, -11.0, -6.0),
                      Decimal("0.30"), f"{affinity:.1f} kcal/mol"))

        return components

    def applicability(self, x) -> Applicability:
        provider = getattr(self.docking_provider, "name", type(self.docking_provider).__name__)
        mock = isinstance(self.docking_provider, MockDockingProvider)
        mol = chem.parse(x.get("smiles", ""))

        note = (
            "Physicochemical descriptors (MW, cLogP, TPSA, H-bond donors/acceptors, "
            "rotatable bonds) and the synthetic-accessibility score are exact and "
            "deterministic from the structure. The binding term is a predicted "
            f"docking affinity from provider '{provider}'."
        )
        if mock:
            note += (
                " MockDockingProvider is a hash-derived mock: the affinity is "
                "illustrative only and carries no physical meaning. Plug Rowan / "
                "Vina (as in the molecule-optimization-sandbox) for a real predicted "
                "affinity."
            )
        else:
            note += " Real docking (Rowan / Vina) predicts affinity, it does not measure it."
        # note when an optional RDKit component was unavailable and thus omitted
        if mol is not None:
            if chem.sa_score(mol) is None:
                note += (
                    " The RDKit Contrib SA_Score module was unavailable at runtime, so "
                    "the synthetic-accessibility term was omitted."
                )
            if not chem.pains_hits(mol):
                # cannot distinguish 'clean' from 'catalog unavailable' after the fact,
                # so flag that the PAINS screen is best-effort on this install.
                note += (
                    " PAINS screening is best-effort: if the RDKit FilterCatalog is "
                    "unavailable, no structural alerts are raised."
                )
        return Applicability(in_domain=True, reason=note)
