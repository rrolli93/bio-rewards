"""A generic, deterministic optimizer that climbs any RewardFunction.

The library gives you the verifier half of RLVR (the deterministic, unforgiving
scorer). This is the other half made watchable: a probabilistic generator that
proposes candidates, a reward function that grades them, and a loop that keeps
what scores higher. The result is a live trajectory where a bad candidate is
iteratively repaired and the reward climbs.

The optimizer is domain-agnostic. It never looks inside ``x`` and never knows
what a primer or a peptide is. It only calls two things: ``reward_fn.evaluate(x)``
to score, and an injected ``mutate(x, rng)`` to propose a neighbour. Point the
same loop at a different reward function and mutation operator and it climbs that
landscape instead.

Determinism is the whole contract of this repo, so it holds here too. All
randomness flows through a single ``random.Random(rng_seed)`` handed to ``mutate``.
No global random, no clock-seeded state. Same seed in, same climb out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from biorewards.core import RewardFunction, RewardResult, quantize

ZERO = Decimal("0")


@dataclass(frozen=True)
class OptimizeResult:
    """The outcome of one optimization run.

    best_candidate  the highest-scoring ``x`` found.
    best_reward     its reward (Decimal in [0, 1]).
    best_result     the full RewardResult for best_candidate (components, gates,
                    applicability), so the caller can inspect why it won.
    trajectory      one record per iteration (iteration 0 = the initial
                    population), each a dict with iteration, best_reward,
                    mean_reward, and best_candidate. best_reward is monotonically
                    non-decreasing because the elite is carried forward unchanged.
    """

    best_candidate: object
    best_reward: Decimal
    best_result: RewardResult
    trajectory: list[dict] = field(default_factory=list)


def _reward_of(result: RewardResult | None) -> Decimal:
    """Reward of an evaluation, treating a rejected or crashed candidate as 0.

    A lethal or invalid mutation (a primer trimmed too short, a peptide that
    trips a hard gate) scores exactly zero rather than crashing the loop. That is
    the same rule the reward skeleton uses: impossible is zero, not an exception.
    """
    if result is None or result.status != "ok":
        return ZERO
    return result.reward


def optimize(
    reward_fn: RewardFunction,
    seed_candidate,
    mutate: Callable,
    iterations: int = 200,
    population: int = 12,
    rng_seed: int = 0,
    elitism: int = 2,
) -> OptimizeResult:
    """Seeded evolutionary hill-climb over any RewardFunction.

    Each round the elite (the top ``elitism`` candidates) survive untouched, and
    their mutated children fill the rest of the population. Everyone is scored,
    the population is re-sorted, and the best is carried into the next round. With
    the elite preserved, the best reward can only stay flat or rise.

    reward_fn   any RewardFunction; scored via reward_fn.evaluate(x).reward.
    seed_candidate  the starting ``x`` (a deliberately poor one makes the climb
                visible).
    mutate      mutate(candidate, rng) -> new_candidate. Domain-specific and
                injected. Must not mutate its input in place.
    iterations  number of generations to run.
    population  candidates kept alive each generation.
    rng_seed    the only source of randomness; fixes the whole trajectory.
    elitism     how many top candidates survive each generation unchanged.
    """
    if population < 1:
        raise ValueError("population must be at least 1")
    elitism = max(1, min(elitism, population))
    rng = random.Random(rng_seed)

    # A monotonically increasing birth index breaks reward ties in favour of the
    # newer candidate. That allows neutral drift: an edit that leaves the reward
    # unchanged can still be kept and built upon, which is how the search crosses
    # flat plateaus (for example a primer whose min-coupled terms need both
    # strands fixed before the score can move). It never lets the best reward
    # drop, because reward is still the primary sort key.
    birth = 0

    def score(candidate) -> RewardResult | None:
        # A broken mutation should degrade to reward 0, never take down the run.
        try:
            return reward_fn.evaluate(candidate)
        except Exception:
            return None

    def rank_key(individual):
        _, result, born = individual
        return (_reward_of(result), born)

    # Initial population: the seed plus mutated copies of it.
    pop = [(seed_candidate, score(seed_candidate), birth)]
    while len(pop) < population:
        birth += 1
        child = mutate(seed_candidate, rng)
        pop.append((child, score(child), birth))
    pop.sort(key=rank_key, reverse=True)

    trajectory: list[dict] = []

    def record(iteration: int) -> None:
        rewards = [_reward_of(res) for _, res, _ in pop]
        mean = sum(rewards, ZERO) / len(rewards)
        best_cand, best_res, _ = pop[0]
        trajectory.append(
            {
                "iteration": iteration,
                "best_reward": _reward_of(best_res),
                "mean_reward": quantize(mean),
                "best_candidate": best_cand,
            }
        )

    record(0)

    for it in range(1, iterations + 1):
        elite = pop[:elitism]
        children = []
        while len(elite) + len(children) < population:
            parent_cand, _, _ = elite[rng.randrange(len(elite))]
            child = mutate(parent_cand, rng)
            birth += 1
            children.append((child, score(child), birth))
        pop = elite + children
        pop.sort(key=rank_key, reverse=True)
        record(it)

    best_candidate, best_result, _ = pop[0]
    return OptimizeResult(
        best_candidate=best_candidate,
        best_reward=_reward_of(best_result),
        best_result=best_result,
        trajectory=trajectory,
    )
