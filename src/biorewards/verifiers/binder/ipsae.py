"""Model-agnostic interface confidence math for binder-target complexes.

The two numbers a protein engineer actually reads off a co-folded complex before
believing an interface: how confident the model is about cross-chain geometry
(pae_interaction) and an interface pTM-style score (ipSAE). Both are computed
here from the Predicted Aligned Error (PAE) matrix alone, which is why they are
folder-agnostic: any co-folder that emits a PAE (Boltz-2, AlphaFold2, Chai) can
back them, and this code never has to change.

The PAE matrix is (N x N), rows/cols ordered [binder residues, then target
residues]. Entry PAE(i, j) is the expected position error (Angstrom) of residue
j when the structure is aligned on residue i. Lower is a more confident pair.

ipSAE is our implementation of the interface Solvation-Accessible pTM score of
Dunbar and Sternberg (2024, "ipSAE: an interface pTM score for scoring
protein-protein interactions from AlphaFold"). The published definition, in the
asymmetric binder->target direction:

  * For each binder residue i, let S_i be the set of target residues j with
    PAE(i, j) < pae_cutoff. Skip i when S_i is empty.
  * n_i = |S_i|.
  * d0_i = max(1.0, 1.24 * (n_i - 15) ** (1/3) - 1.8), the TM-score d0 form. The
    cube root of a negative (n_i < 15) is taken as a real root, then the max
    clamps d0_i to at least 1.0.
  * score_i = mean over j in S_i of  1 / (1 + (PAE(i, j) / d0_i) ** 2).
  * ipSAE(binder->target) = max_i score_i, or 0.0 if no residue qualifies.

Compute ipSAE(target->binder) symmetrically (each target residue i against the
binder residues j). ipsae = max of the two directions. Higher is better, in
[0, 1].

Pure stdlib (math only). The matrices at play here are small, so no numpy.

Sanity doctests. ipSAE rises as the cross-chain PAE falls well below the d0
radius (which clamps at 1.0 A), so a near-zero PAE approaches 1.0 while a PAE
above the cutoff yields 0.

    >>> tiny = [[0.1] * 4 for _ in range(4)]
    >>> m = compute_interface_metrics(tiny, 2, 2)
    >>> round(m["pae_interaction"], 3)
    0.1
    >>> m["ipsae"] > 0.95
    True
    >>> huge = [[29.0] * 4 for _ in range(4)]
    >>> h = compute_interface_metrics(huge, 2, 2)
    >>> round(h["pae_interaction"], 3)
    29.0
    >>> h["ipsae"] < 0.05
    True
"""

from __future__ import annotations

import math


def _cbrt(x: float) -> float:
    """Real cube root, valid for negative x (unlike x ** (1/3))."""
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _d0(n: int) -> float:
    """TM-score radius d0 for n aligned pairs, clamped to at least 1.0."""
    return max(1.0, 1.24 * _cbrt(n - 15) - 1.8)


def _directional_ipsae(pae, rows, cols, pae_cutoff: float) -> float:
    """max over anchor residues i (in rows) of the mean TM-weight to the partner
    residues j (in cols) that fall under the PAE cutoff."""
    best = 0.0
    for i in rows:
        confident = [pae[i][j] for j in cols if pae[i][j] < pae_cutoff]
        if not confident:
            continue
        d0 = _d0(len(confident))
        score_i = sum(1.0 / (1.0 + (e / d0) ** 2) for e in confident) / len(confident)
        if score_i > best:
            best = score_i
    return best


def compute_interface_metrics(pae, n_binder, n_target, pae_cutoff=10.0) -> dict:
    """Interface confidence from a PAE matrix.

    pae is an (N x N) nested sequence (list of lists or similar), N = n_binder +
    n_target, ordered [binder residues, then target residues]. Returns
    {"pae_interaction": float, "ipsae": float}.

      * pae_interaction: mean of the cross-chain PAE block, both the
        binder->target submatrix and the target->binder submatrix. Angstrom,
        lower is better.
      * ipsae: the interface pTM score defined in the module docstring, the max
        of the two asymmetric directions. In [0, 1], higher is better.
    """
    binder = range(0, n_binder)
    target = range(n_binder, n_binder + n_target)

    # pae_interaction: mean over both off-diagonal (cross-chain) blocks.
    cross = [pae[i][j] for i in binder for j in target]
    cross += [pae[i][j] for i in target for j in binder]
    pae_interaction = sum(cross) / len(cross) if cross else 0.0

    fwd = _directional_ipsae(pae, binder, target, pae_cutoff)
    rev = _directional_ipsae(pae, target, binder, pae_cutoff)
    ipsae = max(fwd, rev)

    return {"pae_interaction": pae_interaction, "ipsae": ipsae}
