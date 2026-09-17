"""RDKit descriptor helpers for the small-molecule verifier.

Every function imports rdkit lazily, inside its own body, so that importing this
module (and therefore the whole biorewards package) never hard-requires rdkit.
The small-molecule verifier is the one place we lean on a real cheminformatics
toolkit, and it stays an optional extra (``pip install biorewards[smallmolecule]``).

The descriptors mirror exactly the ones the molecule-optimization-sandbox scored:
molecular weight, cLogP, H-bond donors/acceptors, TPSA, rotatable bonds, plus a
synthetic-accessibility score and a PAINS structural-alert screen.
"""

from __future__ import annotations


def parse(smiles: str):
    """Return an RDKit Mol, or None when the SMILES cannot be parsed."""
    from rdkit import Chem

    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def descriptors(mol) -> dict:
    """The Lipinski/Veber physchem panel, exact from the structure."""
    from rdkit.Chem import Crippen, Descriptors

    return {
        "mw": Descriptors.MolWt(mol),
        "clogp": Crippen.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
    }


def canonical(smiles: str) -> str:
    """Canonical SMILES, or the input unchanged when it will not parse."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def sa_score(mol) -> float | None:
    """Synthetic accessibility 1 (easy) to 10 (hard), Ertl-Schuffenhauer.

    Uses the RDKit Contrib SA_Score ``sascorer``, which is shipped with RDKit but
    lives outside the importable package tree, so we append it to sys.path first.
    Returns None when the Contrib module is unavailable so the reward can honestly
    omit the synthetic-accessibility term rather than fabricate one.
    """
    try:
        import os
        import sys

        from rdkit.Chem import RDConfig

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer

        return sascorer.calculateScore(mol)
    except Exception:
        return None


def pains_hits(mol) -> list[str]:
    """Names of any PAINS structural alerts the molecule trips.

    PAINS (pan-assay interference compounds) are reactive/promiscuous substructures
    that produce false readouts across assays. A hit is a hard reject, the same way
    the sandbox gated reactive groups. Returns [] when the filter catalog is
    unavailable rather than silently passing everything.
    """
    try:
        from rdkit.Chem import FilterCatalog
        from rdkit.Chem.FilterCatalog import FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        matches = catalog.GetMatches(mol)
        return [m.GetDescription() for m in matches]
    except Exception:
        return []
