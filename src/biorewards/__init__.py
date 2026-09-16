"""biorewards -- deterministic verifiable rewards for biology agents.

One skeleton, many verifiers:

    R(x) = 1[hard gates pass] * sum_i  w_i * phi_i( g_i(x) )

See README.md for the framework. Each verifier lives under biorewards.verifiers.
"""

from biorewards.core import (
    Applicability,
    Component,
    RewardFunction,
    RewardResult,
)

__all__ = [
    "RewardFunction",
    "RewardResult",
    "Component",
    "Applicability",
]

__version__ = "0.1.0"
