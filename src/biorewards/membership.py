"""Deterministic membership + calibration functions.

These map a raw physical measurement (a Tm, a docking energy, a GC fraction)
onto the interval [0, 1]. They are the ``phi_i`` of the reward skeleton

    R(x) = 1[hard gates pass] * sum_i  w_i * phi_i( g_i(x) )

Everything is Decimal so the same input yields the same bytes on every machine.
"""

from __future__ import annotations

from decimal import Decimal

ZERO = Decimal("0")
ONE = Decimal("1")


def _d(value: float | int | str | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def clip(value: Decimal) -> Decimal:
    """Clamp to [0, 1]."""
    return max(ZERO, min(ONE, value))


def trapezoid(x, a, b, c, d) -> Decimal:
    """Sweet-spot membership. 0 below a, ramps to 1 on [b, c], back to 0 at d.

    This is the shape that says 'too little is bad AND too much is bad' --
    the biological reality for molecular weight, GC content, melting temp, etc.
    """
    x, a, b, c, d = map(_d, (x, a, b, c, d))
    if x <= a or x >= d:
        return ZERO
    if x < b:
        return clip((x - a) / (b - a))
    if x <= c:
        return ONE
    return clip((d - x) / (d - c))


def falling(x, c, d) -> Decimal:
    """1 up to c, ramps down to 0 at d. 'Smaller is better, past d it's dead.'"""
    x, c, d = map(_d, (x, c, d))
    if x <= c:
        return ONE
    if x >= d:
        return ZERO
    return clip((d - x) / (d - c))


def rising(x, a, b) -> Decimal:
    """0 up to a, ramps up to 1 at b. 'Bigger is better, below a it's dead.'"""
    x, a, b = map(_d, (x, a, b))
    if x <= a:
        return ZERO
    if x >= b:
        return ONE
    return clip((x - a) / (b - a))


def anchored(raw, good, bad) -> Decimal:
    """Calibrated linear map between a good anchor and a bad anchor.

    Works in either direction. For docking energy good=-10, bad=-7 (more
    negative is better); for an off-target count good=0, bad=50. This is
    'calibration before ranking' made literal: the anchors come from a
    reference panel of known positives/negatives, not from the candidate.
    """
    raw, good, bad = map(_d, (raw, good, bad))
    if good == bad:
        return ZERO
    return clip((raw - bad) / (good - bad))
