"""The reward-function contract shared by every biology verifier.

One skeleton, many instances:

    R(x) = 1[hard gates pass] * sum_i  w_i * phi_i( g_i(x) )

A verifier supplies three things:
  * hard_gates(x)      -> binary validity predicates; any failure => reward 0
  * score_components(x)-> the calibrated phi_i * w_i terms
  * applicability(x)   -> whether the candidate is inside the model's domain

The base class does the deterministic bookkeeping: weighted sum, quantization,
a reproducible evaluation key, and an honest "is this in domain" flag. The agent
being scored only ever passes ``x``. It never touches this code. That sealed
boundary is the whole safety story: a probabilistic generator on one side, a
deterministic verifier on the other.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal

QUANTUM = Decimal("0.000001")
ZERO = Decimal("0")
ONE = Decimal("1")


def quantize(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class Component:
    """One calibrated term of the reward. ``raw`` is the physical measurement,
    ``score`` its membership value in [0, 1], ``weight`` its contribution."""

    name: str
    raw: float | None
    score: Decimal
    weight: Decimal
    detail: str = ""


@dataclass(frozen=True)
class Applicability:
    """Does the verifier actually cover this candidate, or is it extrapolating?

    The CYP51 lesson: a reward function that scores outside its validity domain
    without saying so is worse than useless. If ``in_domain`` is False the score
    is advisory and the reason explains what is not being checked.
    """

    in_domain: bool = True
    reason: str = ""


@dataclass(frozen=True)
class RewardResult:
    status: str  # "ok" | "rejected"
    reward: Decimal
    components: tuple[Component, ...]
    hard_gate_failures: tuple[str, ...]
    applicability: Applicability
    evaluation_key: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reward": str(self.reward),
            "components": [
                {
                    "name": c.name,
                    "raw": c.raw,
                    "score": str(c.score),
                    "weight": str(c.weight),
                    "detail": c.detail,
                }
                for c in self.components
            ],
            "hard_gate_failures": list(self.hard_gate_failures),
            "applicability": {
                "in_domain": self.applicability.in_domain,
                "reason": self.applicability.reason,
            },
            "evaluation_key": self.evaluation_key,
            "details": self.details,
        }


class RewardFunction:
    """Base class. Subclass and implement identity/hard_gates/score_components."""

    name: str = "reward"
    #: bump when the scoring logic changes so cached results never collide
    version: str = "0.1.0"

    def config(self) -> dict:
        """Serializable config that defines this scorer. Hashed into the key so
        two runs with the same config + candidate are byte-identical."""
        return {}

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            {"name": self.name, "version": self.version, "config": self.config()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def identity(self, x) -> str:
        """A canonical string for the candidate, used in the evaluation key.
        Override when ``x`` is not a plain string (e.g. a dict of sequences)."""
        return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)

    def _key(self, x) -> str:
        return hashlib.sha256(
            f"{self.manifest_hash}|{self.identity(x)}".encode()
        ).hexdigest()

    # ---- to be implemented by each verifier -----------------------------

    def hard_gates(self, x) -> list[str]:
        """Return a list of failure reasons. Empty list means all gates pass."""
        return []

    def score_components(self, x) -> list[Component]:
        raise NotImplementedError

    def applicability(self, x) -> Applicability:
        return Applicability(in_domain=True)

    # ---- the deterministic machinery ------------------------------------

    def evaluate(self, x) -> RewardResult:
        key = self._key(x)
        failures = tuple(self.hard_gates(x))
        if failures:
            return RewardResult(
                status="rejected",
                reward=quantize(ZERO),
                components=(),
                hard_gate_failures=failures,
                applicability=self.applicability(x),
                evaluation_key=key,
            )
        components = tuple(self.score_components(x))
        total_w = sum((c.weight for c in components), ZERO)
        if total_w == ZERO:
            raw = ZERO
        else:
            raw = sum((c.weight * c.score for c in components), ZERO) / total_w
        return RewardResult(
            status="ok",
            reward=quantize(raw),
            components=components,
            hard_gate_failures=(),
            applicability=self.applicability(x),
            evaluation_key=key,
            details={"total_weight": str(total_w)},
        )
