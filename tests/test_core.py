from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import anchored, falling, rising, trapezoid


def test_trapezoid_shape():
    assert trapezoid(50, 52, 58, 62, 68) == Decimal("0")   # below
    assert trapezoid(60, 52, 58, 62, 68) == Decimal("1")   # plateau
    assert trapezoid(70, 52, 58, 62, 68) == Decimal("0")   # above
    assert Decimal("0") < trapezoid(55, 52, 58, 62, 68) < Decimal("1")  # ramp


def test_anchored_direction_agnostic():
    # docking: more negative is better
    assert anchored(-10, -10, -7) == Decimal("1")
    assert anchored(-7, -10, -7) == Decimal("0")
    # off-target count: smaller is better
    assert anchored(0, 0, 50) == Decimal("1")
    assert anchored(50, 0, 50) == Decimal("0")


def test_rising_falling():
    assert rising(0, 0, 5) == Decimal("0")
    assert rising(5, 0, 5) == Decimal("1")
    assert falling(0, 3, 6) == Decimal("1")
    assert falling(6, 3, 6) == Decimal("0")


class _Toy(RewardFunction):
    name = "toy"

    def hard_gates(self, x):
        return [] if x >= 0 else ["negative"]

    def score_components(self, x):
        return [Component("v", x, trapezoid(x, 0, 1, 2, 3), Decimal("1"))]


def test_hard_gate_zeroes_reward():
    r = _Toy().evaluate(-1)
    assert r.status == "rejected"
    assert r.reward == Decimal("0.000000")


def test_weighted_sum_and_quantization():
    r = _Toy().evaluate(1)  # trapezoid plateau -> 1.0
    assert r.status == "ok"
    assert r.reward == Decimal("1.000000")


def test_manifest_hash_stable_and_config_sensitive():
    class A(RewardFunction):
        name = "a"
        def config(self):
            return {"k": 1}
        def score_components(self, x):
            return [Component("v", 0, Decimal("1"), Decimal("1"))]

    class B(A):
        def config(self):
            return {"k": 2}

    assert A().manifest_hash == A().manifest_hash
    assert A().manifest_hash != B().manifest_hash
