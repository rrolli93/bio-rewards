"""Docking providers for the small-molecule verifier.

Scoring a small molecule for a target means docking it and reading a predicted
binding affinity (kcal/mol, MORE NEGATIVE is better). That computation is
expensive and external, so it is injected as a provider, exactly like the fold
provider on the binder verifier.

``MockDockingProvider`` is a MOCK. Its affinity is a hash of the SMILES mapped
into a plausible range: deterministic, offline, and scientifically meaningless.
It exists to make the reward reproducible and demonstrable with no GPU and no
network. ``RowanDockingProvider`` is the documented real path (it mirrors the
molecule-optimization-sandbox's RowanBindingProvider) and is left as a stub so no
network call is possible from this library.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class DockingProvider(Protocol):
    """Docks a small molecule against the configured target.

    dock(smiles) -> float: predicted binding affinity in kcal/mol, where MORE
    NEGATIVE is better (a stronger predicted binder).
    """

    def dock(self, smiles: str) -> float:
        ...


class MockDockingProvider:
    """MOCK docking provider. Deterministic, offline, scientifically meaningless.

    The pseudo-affinity is sha256(smiles) mapped linearly into [-11.0, -5.0]
    kcal/mol. A given SMILES always docks to the same number, but that number
    carries NO physical meaning: it is here so the reward reproduces and demos with
    no GPU and no network. Real docking is Rowan (as used in the
    molecule-optimization-sandbox) or AutoDock Vina; swap one of those in for an
    affinity that means something.
    """

    name = "mock_docking"

    #: affinity window the hash is mapped into (kcal/mol, more negative is better)
    AFF_MIN = -11.0
    AFF_MAX = -5.0

    def dock(self, smiles: str) -> float:
        digest = hashlib.sha256(smiles.encode()).digest()
        q = int.from_bytes(digest[0:4], "big") / 0xFFFFFFFF  # in [0, 1]
        # q = 1 maps to the strongest (most negative) end of the window
        affinity = self.AFF_MAX + q * (self.AFF_MIN - self.AFF_MAX)
        return round(affinity, 4)


class RowanDockingProvider:
    """Real docking via Rowan (STUB). Mirrors the sandbox's RowanBindingProvider.

    The molecule-optimization-sandbox docked with Rowan's
    ``submit_docking_workflow`` (Vina scoring, credit-guarded, one capped workflow
    per molecule). Wiring that here needs the same things it needed there: a
    ROWAN_API_KEY, a ``protein_uuid`` for the docked receptor, and a pocket box
    (box_center + box_size). Rather than smuggle a billable network call into a
    library that must run offline, ``dock`` raises. Wire the Rowan workflow in a
    separate, credit-aware adapter and inject it as the docking provider.
    """

    name = "rowan_docking"

    def __init__(self, api_key=None, protein_uuid=None, box_center=None, box_size=None):
        self.api_key = api_key
        self.protein_uuid = protein_uuid
        self.box_center = box_center
        self.box_size = box_size

    def dock(self, smiles: str) -> float:
        raise NotImplementedError("wire Rowan submit_docking_workflow separately")
