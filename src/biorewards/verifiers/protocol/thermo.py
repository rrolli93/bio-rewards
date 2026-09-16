"""Deterministic DNA thermodynamics. No ML, no GPU, no network.

The melting temperature is computed with the SantaLucia (1998) unified
nearest-neighbor parameters. This is the exact number an LLM cannot reliably
produce by eye, and the exact number a wet-lab primer either has or does not.
That gap is the whole point: creativity proposes the primer, this verifies it.
"""

from __future__ import annotations

import math

R = 1.987  # cal / (mol K)

# SantaLucia 1998 unified NN parameters: (dH kcal/mol, dS cal/(mol K))
_NN = {
    "AA": (-7.6, -21.3), "TT": (-7.6, -21.3),
    "AT": (-7.2, -20.4),
    "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7), "TG": (-8.5, -22.7),
    "GT": (-8.4, -22.4), "AC": (-8.4, -22.4),
    "CT": (-7.8, -21.0), "AG": (-7.8, -21.0),
    "GA": (-8.2, -22.2), "TC": (-8.2, -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9), "CC": (-8.0, -19.9),
}
_INIT_GC = (0.1, -2.8)   # initiation with terminal G or C
_INIT_AT = (2.3, 4.1)    # initiation with terminal A or T

_COMP = str.maketrans("ACGT", "TGCA")


def is_dna(seq: str) -> bool:
    return len(seq) > 0 and set(seq.upper()) <= set("ACGT")


def gc_content(seq: str) -> float:
    """GC fraction as a percentage."""
    s = seq.upper()
    return 100.0 * (s.count("G") + s.count("C")) / len(s)


def revcomp(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


def tm_nn(seq: str, strand_conc_M: float = 0.25e-6, na_M: float = 0.05) -> float:
    """Nearest-neighbor Tm in Celsius for a non-self-complementary duplex.

    strand_conc_M: total oligo strand concentration (default 250 nM).
    na_M: monovalent cation concentration for the salt correction.
    """
    s = seq.upper()
    if len(s) < 2:
        raise ValueError("sequence too short for a nearest-neighbor Tm")

    dH, dS = 0.0, 0.0
    # terminal initiation on both ends
    for end in (s[0], s[-1]):
        ih, is_ = _INIT_GC if end in "GC" else _INIT_AT
        dH += ih
        dS += is_
    # nearest-neighbor stack sum
    for i in range(len(s) - 1):
        h, sderiv = _NN[s[i:i + 2]]
        dH += h
        dS += sderiv

    # SantaLucia 1998 salt correction on entropy
    dS = dS + 0.368 * (len(s) - 1) * math.log(na_M)

    # non-self-complementary: divide strand conc by 4
    tm_kelvin = (1000.0 * dH) / (dS + R * math.log(strand_conc_M / 4.0))
    return tm_kelvin - 273.15


def three_prime_gc_clamp(seq: str, window: int = 5) -> int:
    """Count G/C in the last ``window`` bases. A clamp of 1-3 is ideal; a run of
    G/C longer than that promotes mispriming."""
    tail = seq.upper()[-window:]
    return tail.count("G") + tail.count("C")


def max_self_complement_run(seq: str) -> int:
    """Longest run where the 3' end folds back on itself (a hairpin/dimer proxy).

    Deterministic and cheap: slide the reverse complement against the sequence
    and return the longest contiguous complementary stretch. This is a heuristic,
    not a full partition-function fold, and the verifier flags it as such.
    """
    s = seq.upper()
    rc = revcomp(s)
    best = 0
    for offset in range(-(len(s) - 1), len(s)):
        run = 0
        for i in range(len(s)):
            j = i + offset
            if 0 <= j < len(rc) and s[i] == rc[j]:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best
