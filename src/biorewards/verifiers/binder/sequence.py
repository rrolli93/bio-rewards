"""Deterministic peptide/protein sequence biophysics. No ML, no GPU, no network.

These are the developability numbers a designed binder either has or does not,
computed from the sequence alone. They are the exact quantities an LLM cannot
produce reliably by eye (a GRAVY score, a net charge at pH 7.4, a hydrophobic
patch length) and the exact quantities that decide whether a sequence expresses,
stays soluble, and survives a Pichia secretion path. Creativity proposes the
binder, this verifies it.
"""

from __future__ import annotations

import math

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy scale.
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Residues counted as hydrophobic for the patch/run heuristic (AILMFVWY).
_HYDROPHOBIC = set("AILMFVWY")

# Standard side-chain pKa values plus the two termini.
_PKA_POS = {"K": 10.53, "R": 12.48, "H": 6.0}       # basic (protonated = +1)
_PKA_NEG = {"D": 3.65, "E": 4.25, "C": 8.3, "Y": 10.07}  # acidic (deprotonated = -1)
_PKA_NTERM = 9.0
_PKA_CTERM = 2.0


def is_protein(seq: str) -> bool:
    return len(seq) > 0 and set(seq.upper()) <= CANONICAL


def non_standard_residues(seq: str) -> set[str]:
    """The set of characters that are not one of the 20 canonical amino acids."""
    return set(seq.upper()) - CANONICAL


def gravy(seq: str) -> float:
    """Grand average of hydropathy (Kyte-Doolittle mean). Positive is hydrophobic
    (aggregation risk); mildly negative is the soluble sweet spot."""
    s = seq.upper()
    return sum(_KD[a] for a in s) / len(s)


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge at a given pH via Henderson-Hasselbalch, termini included.

    A basic group carries fractional positive charge 1/(1 + 10^(pH - pKa));
    an acidic group carries fractional negative charge -1/(1 + 10^(pKa - pH)).
    """
    s = seq.upper()
    charge = 0.0
    # N-terminus (basic) and C-terminus (acidic)
    charge += 1.0 / (1.0 + 10 ** (ph - _PKA_NTERM))
    charge -= 1.0 / (1.0 + 10 ** (_PKA_CTERM - ph))
    for aa, pka in _PKA_POS.items():
        charge += s.count(aa) * (1.0 / (1.0 + 10 ** (ph - pka)))
    for aa, pka in _PKA_NEG.items():
        charge -= s.count(aa) * (1.0 / (1.0 + 10 ** (pka - ph)))
    return charge


def aromatic_fraction(seq: str) -> float:
    """Fraction of F + W + Y residues."""
    s = seq.upper()
    return (s.count("F") + s.count("W") + s.count("Y")) / len(s)


def cysteine_count(seq: str) -> int:
    return seq.upper().count("C")


def cysteine_parity_score(seq: str) -> float:
    """1.0 when the cysteine count is even (including zero), lower when odd.

    An odd number of cysteines leaves an unpaired thiol (a disulfide-scrambling
    and aggregation liability). This is a deterministic penalty, not a ramp.
    """
    return 1.0 if cysteine_count(seq) % 2 == 0 else 0.3


def kex2_sites(seq: str) -> int:
    """Count KR and RR dibasic motifs (yeast/Pichia Kex2 protease cleavage).

    Each such motif is a place a secreted construct can be clipped in half.
    """
    s = seq.upper()
    count = 0
    for i in range(len(s) - 1):
        pair = s[i:i + 2]
        if pair in ("KR", "RR"):
            count += 1
    return count


def longest_hydrophobic_run(seq: str) -> int:
    """Longest contiguous run of hydrophobic residues (AILMFVWY).

    A long exposed hydrophobic patch is the classic aggregation hot spot.
    """
    s = seq.upper()
    best = run = 0
    for aa in s:
        if aa in _HYDROPHOBIC:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best
