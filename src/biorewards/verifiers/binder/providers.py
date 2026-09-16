"""Fold-confidence providers for the binder verifier.

The developability terms of the binder reward are exact from the sequence. The
structural-confidence terms (pLDDT, interface PAE, design->refold RMSD) are not:
they come from a folding model (ESMFold, AlphaFold2, Boltz). Those live behind a
``FoldProvider`` so the reward logic stays pure, offline, and testable, and so a
real model drops in without touching the scorer.

``MockFoldProvider`` is a MOCK. It fabricates the three numbers deterministically
from a hash of the sequence so the same sequence always yields the same values.
It exists only to make the reward reproducible and demonstrable with no GPU and
no network. Its numbers are illustrative and carry no structural meaning. Plug a
real provider for real confidence.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class FoldProvider(Protocol):
    """Returns fold-confidence metrics for a sequence.

    fold(sequence) -> dict with keys:
      * plddt                 predicted local distance difference test, [0, 100], higher better
      * pae_interaction       predicted aligned error at the interface, Angstrom, lower better
      * self_consistency_rmsd design->refold backbone RMSD, Angstrom, lower better
    """

    def fold(self, sequence: str) -> dict:
        ...


class MockFoldProvider:
    """MOCK fold provider. Deterministic, offline, and scientifically meaningless.

    The three metrics are derived from independent slices of the SHA-256 digest of
    the (uppercased) sequence, then mapped into plausible ranges:
      * plddt                 [40, 95]
      * pae_interaction       [3, 28]
      * self_consistency_rmsd [0.5, 6.0]

    Same sequence in, same numbers out, on every machine. This lets the binder
    reward run and be tested with no GPU and no network. Swap in ESMFold / AF2 /
    Boltz for values that mean something.
    """

    name = "mock_fold"

    def fold(self, sequence: str) -> dict:
        digest = hashlib.sha256(sequence.upper().encode()).digest()
        # three independent unsigned integers from disjoint byte slices
        u_plddt = int.from_bytes(digest[0:4], "big") / 0xFFFFFFFF
        u_pae = int.from_bytes(digest[4:8], "big") / 0xFFFFFFFF
        u_rmsd = int.from_bytes(digest[8:12], "big") / 0xFFFFFFFF
        return {
            "plddt": round(40.0 + u_plddt * (95.0 - 40.0), 3),
            "pae_interaction": round(3.0 + u_pae * (28.0 - 3.0), 3),
            "self_consistency_rmsd": round(0.5 + u_rmsd * (6.0 - 0.5), 3),
        }
