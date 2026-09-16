"""Domain mutation operators for the optimization loop.

A mutator is the generator half of the loop: given a candidate and a seeded RNG,
it proposes one neighbour. It knows the alphabet and the shape of its domain (DNA
primers, protein sequences) but nothing about the reward. That separation keeps
the optimizer generic and the scoring sealed.

Every operator here is pure: it reads the input candidate and returns a new one,
never mutating the argument in place (so the elite carried across generations
stays intact). Given the same RNG state, it makes the same move.
"""

from __future__ import annotations

import random

BASES = "ACGT"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def dna_primer_pair_mutator(
    candidate: dict, rng: random.Random, min_len: int = 15, max_len: int = 30
) -> dict:
    """Mutate one strand of a {'fwd', 'rev'} primer pair by one small edit.

    Picks a strand, then one of: point substitution, extend by one base, or trim
    by one base. Trimming respects ``min_len`` and extension respects ``max_len``
    so the pair does not run away to a degenerate length. Uses only ACGT. Returns
    a new dict; the input is left untouched.
    """
    out = {"fwd": candidate["fwd"], "rev": candidate["rev"]}
    role = rng.choice(("fwd", "rev"))
    seq = out[role]

    ops = ["sub"]
    if len(seq) < max_len:
        ops.append("extend")
    if len(seq) > min_len:
        ops.append("trim")
    op = rng.choice(ops)

    if op == "sub":
        i = rng.randrange(len(seq))
        seq = seq[:i] + rng.choice(BASES) + seq[i + 1:]
    elif op == "extend":
        # grow at a randomly chosen end
        if rng.random() < 0.5:
            seq = rng.choice(BASES) + seq
        else:
            seq = seq + rng.choice(BASES)
    else:  # trim
        if rng.random() < 0.5:
            seq = seq[1:]
        else:
            seq = seq[:-1]

    out[role] = seq
    return out


def protein_mutator(
    candidate: dict, rng: random.Random, min_len: int = 8, max_len: int = 150
) -> dict:
    """Mutate a {'sequence': ...} protein by one point substitution, insertion,
    or deletion over the 20 canonical amino acids.

    Length is held inside [``min_len``, ``max_len``]: deletion is only offered
    above the floor, insertion only below the ceiling. Any other keys on the
    candidate (for example 'target') are preserved. Returns a new dict.
    """
    seq = candidate["sequence"]

    ops = ["sub"]
    if len(seq) < max_len:
        ops.append("ins")
    if len(seq) > min_len:
        ops.append("del")
    op = rng.choice(ops)

    if op == "sub":
        i = rng.randrange(len(seq))
        seq = seq[:i] + rng.choice(AMINO_ACIDS) + seq[i + 1:]
    elif op == "ins":
        i = rng.randrange(len(seq) + 1)
        seq = seq[:i] + rng.choice(AMINO_ACIDS) + seq[i:]
    else:  # del
        i = rng.randrange(len(seq))
        seq = seq[:i] + seq[i + 1:]

    out = dict(candidate)
    out["sequence"] = seq
    return out
