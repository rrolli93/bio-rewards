"""Reward functions for molecular-biology protocol design.

PrimerPairReward  -- score a PCR primer pair the way a bench scientist would.
GuideReward       -- score a CRISPR guide for its deterministic constraints.

Both are pure, offline, and reproducible. They catch the failure LLMs make most:
confidently emitting an oligo whose Tm, GC, PAM, or self-complementarity is wrong.
"""

from __future__ import annotations

from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import anchored, falling, rising, trapezoid

from . import thermo


class PrimerPairReward(RewardFunction):
    name = "primer_pair"
    version = "0.1.0"

    def __init__(
        self,
        tm_ideal=(58, 62),
        tm_edges=(52, 68),
        gc_ideal=(45, 55),
        gc_edges=(35, 65),
        len_range=(18, 30),
        na_M=0.05,
        strand_conc_M=0.25e-6,
    ):
        self.tm_ideal = tm_ideal
        self.tm_edges = tm_edges
        self.gc_ideal = gc_ideal
        self.gc_edges = gc_edges
        self.len_range = len_range
        self.na_M = na_M
        self.strand_conc_M = strand_conc_M

    def config(self) -> dict:
        return {
            "tm_ideal": self.tm_ideal,
            "tm_edges": self.tm_edges,
            "gc_ideal": self.gc_ideal,
            "gc_edges": self.gc_edges,
            "len_range": self.len_range,
            "na_M": self.na_M,
            "strand_conc_M": self.strand_conc_M,
        }

    def identity(self, x) -> str:
        return f"{x['fwd'].upper()}|{x['rev'].upper()}"

    def hard_gates(self, x) -> list[str]:
        fails = []
        for role in ("fwd", "rev"):
            seq = x.get(role, "")
            if not seq:
                fails.append(f"{role}: empty")
                continue
            if not thermo.is_dna(seq):
                fails.append(f"{role}: non-ACGT characters")
            lo, hi = self.len_range
            if not (lo <= len(seq) <= hi):
                fails.append(f"{role}: length {len(seq)} outside [{lo}, {hi}]")
        return fails

    def _tm(self, seq):
        return thermo.tm_nn(seq, self.strand_conc_M, self.na_M)

    def score_components(self, x) -> list[Component]:
        fwd, rev = x["fwd"].upper(), x["rev"].upper()
        tm_f, tm_r = self._tm(fwd), self._tm(rev)
        gc_f, gc_r = thermo.gc_content(fwd), thermo.gc_content(rev)
        a, b = self.tm_edges[0], self.tm_ideal[0]
        c, d = self.tm_ideal[1], self.tm_edges[1]
        ga, gb = self.gc_edges[0], self.gc_ideal[0]
        gc_, gd = self.gc_ideal[1], self.gc_edges[1]

        clamp_f = thermo.three_prime_gc_clamp(fwd)
        clamp_r = thermo.three_prime_gc_clamp(rev)
        dimer = max(thermo.max_self_complement_run(fwd), thermo.max_self_complement_run(rev))

        return [
            Component("tm_fwd", round(tm_f, 2), trapezoid(tm_f, a, b, c, d), Decimal("0.20"),
                      f"{tm_f:.1f}C"),
            Component("tm_rev", round(tm_r, 2), trapezoid(tm_r, a, b, c, d), Decimal("0.20"),
                      f"{tm_r:.1f}C"),
            Component("tm_match", round(abs(tm_f - tm_r), 2), falling(abs(tm_f - tm_r), 1, 5),
                      Decimal("0.20"), f"dTm {abs(tm_f - tm_r):.1f}C"),
            Component("gc_fwd", round(gc_f, 1), trapezoid(gc_f, ga, gb, gc_, gd), Decimal("0.10"),
                      f"{gc_f:.0f}%"),
            Component("gc_rev", round(gc_r, 1), trapezoid(gc_r, ga, gb, gc_, gd), Decimal("0.10"),
                      f"{gc_r:.0f}%"),
            # a 3' clamp of 1-3 G/C is ideal; 0 is weak, 4-5 promotes mispriming
            Component("gc_clamp", min(clamp_f, clamp_r),
                      trapezoid(min(clamp_f, clamp_r), 0, 1, 3, 5), Decimal("0.10"),
                      f"min 3' G/C={min(clamp_f, clamp_r)}"),
            Component("self_complement", dimer, falling(dimer, 4, 10), Decimal("0.10"),
                      f"max run={dimer}"),
        ]

    def applicability(self, x) -> Applicability:
        return Applicability(
            in_domain=True,
            reason="Tm is exact (SantaLucia98 NN). Dimer/hairpin is a run-length "
            "heuristic; specificity vs a genome needs an alignment provider.",
        )


class GuideReward(RewardFunction):
    """CRISPR-Cas9 guide. Expects x = {'guide': 20mer, 'pam': 'NGG-context'}.

    ``pam`` is the 3 nt immediately 3' of the protospacer. The on-target term
    here is a deterministic composition proxy; a calibrated Doench Rule-Set-2
    score is the drop-in provider upgrade and is noted in applicability.
    """

    name = "crispr_guide"
    version = "0.1.0"

    def __init__(self, length=20, gc_ideal=(40, 70), gc_edges=(20, 85)):
        self.length = length
        self.gc_ideal = gc_ideal
        self.gc_edges = gc_edges

    def config(self) -> dict:
        return {"length": self.length, "gc_ideal": self.gc_ideal, "gc_edges": self.gc_edges}

    def identity(self, x) -> str:
        return f"{x['guide'].upper()}|{x.get('pam', '').upper()}"

    def hard_gates(self, x) -> list[str]:
        fails = []
        guide = x.get("guide", "")
        if not thermo.is_dna(guide):
            fails.append("guide: non-ACGT")
        elif len(guide) != self.length:
            fails.append(f"guide: length {len(guide)} != {self.length}")
        pam = x.get("pam", "").upper()
        # Cas9 canonical PAM = NGG. No PAM => the guide cannot be cut, full stop.
        if len(pam) < 2 or pam[-2:] != "GG":
            fails.append(f"pam: '{pam}' is not NGG")
        # a run of >=4 T is a Pol III terminator for U6-driven guides
        if "TTTT" in guide.upper():
            fails.append("guide: poly-T (TTTT) terminator")
        return fails

    def score_components(self, x) -> list[Component]:
        guide = x["guide"].upper()
        gc = thermo.gc_content(guide)
        ga, gb = self.gc_edges[0], self.gc_ideal[0]
        gc_, gd = self.gc_ideal[1], self.gc_edges[1]
        homopol = _longest_homopolymer(guide)
        return [
            Component("gc", round(gc, 1), trapezoid(gc, ga, gb, gc_, gd), Decimal("0.5"),
                      f"{gc:.0f}%"),
            # long homopolymer runs hurt synthesis and activity
            Component("no_homopolymer", homopol, falling(homopol, 3, 6), Decimal("0.5"),
                      f"longest run={homopol}"),
        ]

    def applicability(self, x) -> Applicability:
        return Applicability(
            in_domain=True,
            reason="PAM + terminator + composition are exact. On-target efficiency "
            "is a proxy; Doench RS2 and genome-wide off-target CFD are provider upgrades.",
        )


def _longest_homopolymer(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best
