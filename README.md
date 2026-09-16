# biorewards — Verifiable Rewards for Biology

**A deterministic verifier layer for probabilistic biology agents.**

Reasoning models got good at math and code because correctness there is *cheap
to check*: a proof verifier, a unit test, a compiler. That closed loop has a
name, RLVR (reinforcement learning with verifiable rewards). The generator is
probabilistic and creative; the verifier is deterministic and unforgiving; the
seam between them is where the learning happens.

Biology mostly lacks those verifiers, so bio-agents drift and hallucinate with
confidence. `biorewards` is a small library of them. Hallucination is a feature
when you are generating hypotheses. It is a defect in any workflow that needs
precision. This library is the precision half.

## The one skeleton

Every verifier is an instance of the same reward function:

```
R(x) = 1[ hard gates pass ] · Σ_i  w_i · φ_i( g_i(x) )
```

- `x` — the artifact the agent proposes. It only ever passes `x`. It never
  touches the scoring code. That sealed boundary is the safety story.
- **hard gates** — binary validity predicates. Anything physically or
  biologically impossible scores exactly zero. No partial credit.
- `g_i(x)` — a deterministic *measurement* (a melting temperature, a docking
  energy, a metabolic flux, a conservation score).
- `φ_i` — a calibrated membership function mapping that measurement to `[0,1]`.
  Trapezoids encode biological sweet spots: too little is bad *and* too much is
  bad.
- `w_i` — weights. Multi-objective scalarization.

## The five properties that make a reward function trustworthy

1. **Deterministic.** `Decimal` arithmetic, fixed rounding, hashed config. Same
   input gives the same bytes on every machine, so results cache and reproduce.
2. **Calibrated.** Membership anchors come from a reference panel of known
   positives and negatives, set *before* any novel candidate is scored.
3. **Hard gates separate from the graded score.** Validity is not a soft
   penalty. Impossible is zero.
4. **Honest applicability domain.** Every result reports whether the candidate
   is inside the verifier's competence. The origin lesson: a small-molecule
   docking reward scored a drug-resistant enzyme identically to wild type
   because rigid docking is blind to the resistance mechanism. A reward function
   that extrapolates silently is worse than none.
5. **Goodhart-resistant.** Hard gates plus multiple objectives plus adversarial
   negative controls (scrambles, decoys) make the metric expensive to cheat.

## The provider pattern

Expensive or external computations (docking, folding, an FBA solve, a DOI
lookup) are injected as *providers*, each with a mock fallback. The reward logic
stays pure and testable offline, and every demo runs at a venue with no GPU and
no network.

## The four verifiers

| Domain | Artifact `x` | Deterministic ground truth |
|---|---|---|
| **protocol** | PCR primer pair, CRISPR guide, cloning junction | nearest-neighbor Tm, GC, PAM, self-complementarity |
| **binder** | designed protein / peptide sequence | fold self-consistency (RMSD, pLDDT, ipSAE) + developability gates |
| **metabolic** | gene knockout / insertion set | flux balance analysis (a linear program) |
| **citation** | a factual scientific claim | DOI resolves, numbers match source, entity type correct |

`protocol` is implemented and fully offline. The other three ship next, each as
a subclass of the same `RewardFunction`.

## Quick start

```python
from biorewards.verifiers.protocol import PrimerPairReward

rf = PrimerPairReward()
result = rf.evaluate({"fwd": "ACCACAGTCCATGCCATCAC",
                      "rev": "TCCACCACCCTGTTGCTGTA"})
print(result.reward)                 # Decimal in [0,1], deterministic
print(result.applicability.reason)   # what is and is not being checked
for c in result.components:
    print(c.name, c.raw, c.score, c.weight)
```

## Layout

```
src/biorewards/
  core.py          # RewardFunction, RewardResult, the deterministic machinery
  membership.py    # trapezoid / rising / falling / anchored  (the φ_i)
  verifiers/
    protocol/      # domain 1: primers, guides, cloning  (implemented)
    binder/        # domain 2  (next)
    metabolic/     # domain 3  (next)
    citation/      # domain 4  (next)
tests/
```

## Test

```
uv run --with pytest python -m pytest -q
```

Built for the NVIDIA × OpenAI life-sciences hackathon, London, 2026-09-18.
