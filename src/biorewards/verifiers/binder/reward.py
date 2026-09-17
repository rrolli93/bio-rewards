"""Reward function for designed protein / peptide binders.

BinderReward -- score a designed binder the way a protein engineer would before
spending a wet-lab slot on it. Two families of terms:

  * developability, computed exactly from the sequence (GRAVY, net charge,
    aromatic fraction, cysteine parity, Kex2 sites, hydrophobic patch)
  * fold confidence, from an injected FoldProvider. When a target is given the
    provider co-folds the binder-target COMPLEX and we score the interface with
    the field-standard set: ipTM, ipSAE, pae_interaction, plus binder pLDDT.
    With no target we can only fold the monomer, so we score pLDDT alone and omit
    the three interface terms.

The developability half is deterministic and offline. The fold half is only as
real as the provider: with MockFoldProvider it is illustrative, with AF2 / Boltz
/ Chai it is structural confidence. ipSAE and pae_interaction are computed by us
from the provider's PAE matrix, so the folder is swappable. Applicability says
which is which.
"""

from __future__ import annotations

from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import anchored, falling, trapezoid

from . import sequence
from .ipsae import compute_interface_metrics
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
        # target folds into the identity so a binder scored alone and the same
        # binder scored against a target cache as distinct complexes.
        binder = x.get("sequence", "").upper()
        target = (x.get("target") or "").upper()
        return f"{binder}|{target}"

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
        # A target means co-fold the complex and score the interface; no target
        # means monomer only, so pLDDT is the one fold term we can honestly emit.
        fold = self.fold_provider.fold(seq, x.get("target"))
        plddt = fold["plddt"]

        components = [
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
        ]

        # Interface terms only exist for a co-folded complex.
        if fold["n_target"] > 0 and fold["iptm"] is not None:
            iptm = fold["iptm"]
            components.append(
                Component("iptm", round(iptm, 3), anchored(iptm, 0.85, 0.30), Decimal("0.10"),
                          f"{iptm:.2f}"))
            # ipSAE and pae_interaction need the PAE matrix. Some folders (e.g.
            # LiteFold) return ipTM and pLDDT but not PAE; then we honestly score
            # ipTM alone and skip the two PAE-derived terms.
            pae_matrix = fold.get("pae")
            if pae_matrix is not None:
                m = compute_interface_metrics(pae_matrix, fold["n_binder"], fold["n_target"])
                ipsae = m["ipsae"]
                pae = m["pae_interaction"]
                components += [
                    # ipSAE: our implementation of the Dunbar-Sternberg interface pTM
                    Component("ipsae", round(ipsae, 3), anchored(ipsae, 0.80, 0.20),
                              Decimal("0.15"), f"{ipsae:.2f}"),
                    # interface PAE: the bold-generative-engine binder filter (gate <10)
                    Component("pae_interaction", round(pae, 2), anchored(pae, 5, 25),
                              Decimal("0.10"), f"{pae:.1f} A"),
                ]

        return components

    def applicability(self, x) -> Applicability:
        provider = getattr(self.fold_provider, "name", type(self.fold_provider).__name__)
        mock = isinstance(self.fold_provider, MockFoldProvider)
        has_target = bool(x.get("target"))
        note = (
            "Developability terms (GRAVY, net charge, aromatic fraction, cysteine "
            "parity, Kex2 sites, hydrophobic patch) are exact from the sequence. "
            "pLDDT and the interface-confidence terms (ipTM, ipSAE, pae_interaction) "
            f"come from co-folding the complex via provider '{provider}'. ipSAE and "
            "pae_interaction are computed by us from the PAE matrix, so the folder is "
            "swappable (Boltz-2, AF2, Chai)."
        )
        if not has_target:
            note += (
                " No target was given, so only the monomer was folded: the interface "
                "terms (ipTM, ipSAE, pae_interaction) were skipped and pLDDT is the "
                "one fold term scored."
            )
        if mock:
            note += (
                " MockFoldProvider is a hash-derived mock: the fold terms are "
                "illustrative only (the PAE input is fabricated, though the ipSAE "
                "computation on it is real). Plug Boltz-2 / AF2 / Chai for real "
                "structural confidence."
            )
        return Applicability(in_domain=True, reason=note)
