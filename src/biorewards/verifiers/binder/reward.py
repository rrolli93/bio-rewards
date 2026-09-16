"""Reward function for designed protein / peptide binders.

BinderReward -- score a designed binder the way a protein engineer would before
spending a wet-lab slot on it. Two families of terms:

  * developability, computed exactly from the sequence (GRAVY, net charge,
    aromatic fraction, cysteine parity, Kex2 sites, hydrophobic patch)
  * fold confidence, from an injected FoldProvider (pLDDT, interface PAE,
    design->refold self-consistency RMSD)

The developability half is deterministic and offline. The fold half is only as
real as the provider: with MockFoldProvider it is illustrative, with ESMFold /
AF2 / Boltz it is structural confidence. Applicability says which is which.
"""

from __future__ import annotations

from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import anchored, falling, trapezoid

from . import sequence
from .providers import FoldProvider, MockFoldProvider


class BinderReward(RewardFunction):
    """Designed binder. Expects x = {'sequence': '<amino acids>', 'target': '<name>'}.

    ``target`` is optional and advisory (the developability terms do not depend on
    it). identity(x) is the uppercased sequence, so the score caches by sequence.
    """

    name = "binder"
    version = "0.1.0"

    def __init__(
        self,
        fold_provider: FoldProvider | None = None,
        len_range=(8, 150),
        gravy_ideal=(-0.8, 0.2),
        gravy_edges=(-1.5, 0.8),
        charge_ideal=(-2, 6),
        charge_edges=(-8, 12),
        aromatic_ideal=(0.05, 0.15),
        aromatic_edges=(0.0, 0.30),
    ):
        self.fold_provider = fold_provider or MockFoldProvider()
        self.len_range = len_range
        self.gravy_ideal = gravy_ideal
        self.gravy_edges = gravy_edges
        self.charge_ideal = charge_ideal
        self.charge_edges = charge_edges
        self.aromatic_ideal = aromatic_ideal
        self.aromatic_edges = aromatic_edges

    def config(self) -> dict:
        return {
            "fold_provider": getattr(self.fold_provider, "name", type(self.fold_provider).__name__),
            "len_range": self.len_range,
            "gravy_ideal": self.gravy_ideal,
            "gravy_edges": self.gravy_edges,
            "charge_ideal": self.charge_ideal,
            "charge_edges": self.charge_edges,
            "aromatic_ideal": self.aromatic_ideal,
            "aromatic_edges": self.aromatic_edges,
        }

    def identity(self, x) -> str:
        return x.get("sequence", "").upper()

    def hard_gates(self, x) -> list[str]:
        fails = []
        seq = x.get("sequence", "")
        if not seq:
            fails.append("sequence: empty")
            return fails
        nonstd = sequence.non_standard_residues(seq)
        if nonstd:
            fails.append(f"sequence: non-standard residues {sorted(nonstd)}")
        lo, hi = self.len_range
        if not (lo <= len(seq) <= hi):
            fails.append(f"sequence: length {len(seq)} outside [{lo}, {hi}]")
        return fails

    def score_components(self, x) -> list[Component]:
        seq = x["sequence"].upper()

        # ---- developability (exact from sequence) ----
        gv = sequence.gravy(seq)
        ga, gb = self.gravy_edges[0], self.gravy_ideal[0]
        gc_, gd = self.gravy_ideal[1], self.gravy_edges[1]

        charge = sequence.net_charge(seq)
        ca, cb = self.charge_edges[0], self.charge_ideal[0]
        cc, cd = self.charge_ideal[1], self.charge_edges[1]

        arom = sequence.aromatic_fraction(seq)
        aa, ab = self.aromatic_edges[0], self.aromatic_ideal[0]
        ac, ad = self.aromatic_ideal[1], self.aromatic_edges[1]

        n_cys = sequence.cysteine_count(seq)
        kex2 = sequence.kex2_sites(seq)
        patch = sequence.longest_hydrophobic_run(seq)

        # ---- fold confidence (from the injected provider) ----
        fold = self.fold_provider.fold(seq)
        plddt = fold["plddt"]
        pae = fold["pae_interaction"]
        rmsd = fold["self_consistency_rmsd"]

        return [
            Component("gravy", round(gv, 3), trapezoid(gv, ga, gb, gc_, gd), Decimal("0.15"),
                      f"{gv:.2f}"),
            Component("net_charge", round(charge, 2), trapezoid(charge, ca, cb, cc, cd),
                      Decimal("0.10"), f"{charge:+.1f} at pH7.4"),
            Component("aromatic_fraction", round(arom, 3), trapezoid(arom, aa, ab, ac, ad),
                      Decimal("0.05"), f"{arom:.0%}"),
            # odd cysteine count => unpaired thiol liability
            Component("cysteine_parity", n_cys,
                      Decimal(str(sequence.cysteine_parity_score(seq))), Decimal("0.10"),
                      f"{n_cys} Cys ({'even' if n_cys % 2 == 0 else 'ODD'})"),
            # KR/RR dibasic motifs get clipped by Kex2 on a secretion path
            Component("kex2_sites", kex2, falling(kex2, 0, 3), Decimal("0.10"),
                      f"{kex2} dibasic"),
            Component("hydrophobic_patch", patch, falling(patch, 4, 9), Decimal("0.10"),
                      f"longest run={patch}"),
            Component("plddt", round(plddt, 2), anchored(plddt, 90, 50), Decimal("0.10"),
                      f"{plddt:.0f}"),
            # interface PAE: the bold-generative-engine binder filter (real gate <10)
            Component("pae_interaction", round(pae, 2), anchored(pae, 5, 25), Decimal("0.15"),
                      f"{pae:.1f} A"),
            Component("self_consistency_rmsd", round(rmsd, 2), anchored(rmsd, 1.0, 5.0),
                      Decimal("0.05"), f"{rmsd:.2f} A"),
        ]

    def applicability(self, x) -> Applicability:
        provider = getattr(self.fold_provider, "name", type(self.fold_provider).__name__)
        mock = isinstance(self.fold_provider, MockFoldProvider)
        note = (
            "Developability terms (GRAVY, net charge, aromatic fraction, cysteine "
            "parity, Kex2 sites, hydrophobic patch) are exact from the sequence. "
            "Fold-confidence terms (pLDDT, interface PAE, self-consistency RMSD) "
            f"come from provider '{provider}'."
        )
        if mock:
            note += (
                " MockFoldProvider is a hash-derived mock: those three terms are "
                "illustrative only. Plug ESMFold / AF2 / Boltz for real structural "
                "confidence."
            )
        return Applicability(in_domain=True, reason=note)
