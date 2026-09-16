"""Live DOI + PDB resolution via LiveResolver (Crossref + RCSB).

Run with:

    uv run --with requests python scripts/citation_live.py

Resolves a real DOI through Crossref and two real PDB ids through RCSB, then
scores a real claim with CitationReward wired to the live resolver. Needs network.
"""

from biorewards.verifiers.citation import CitationReward
from biorewards.verifiers.citation.providers import LiveResolver

# NumPy paper. Crossref carries a JATS-tagged abstract, so it exercises the tag
# stripping and gives real content for the claim-support proxy to score against.
DOI = "10.1038/s41586-020-2649-2"


def main():
    resolver = LiveResolver()

    print("Crossref DOI resolution")
    meta = resolver.resolve_doi(DOI)
    if meta is None:
        print(f"  {DOI} did not resolve (network?).")
        return
    print(f"  doi:      {DOI}")
    print(f"  title:    {meta['title']}")
    print(f"  year:     {meta['year']}")
    print(f"  abstract: {meta['abstract'][:160]}...")
    print()

    print("RCSB PDB entity resolution")
    for pdb in ("3S36", "5EHF"):
        print(f"  {pdb}: {resolver.entity_identity('PDB', pdb)}")
    print()

    # Real claim grounded in the resolved paper. No fabricated entity, no fake
    # number; the DOI resolves and the claim overlaps the source content.
    claim = {
        "claim": "NumPy is the primary array programming library for the Python "
        "language, operating on vectors and matrices.",
        "doi": DOI,
    }
    print("CitationReward with LiveResolver on a real claim")
    result = CitationReward(resolver=LiveResolver()).evaluate(claim)
    print(f"  status: {result.status}")
    print(f"  reward: {result.reward}")
    for c in result.components:
        print(f"    {c.name}: raw={c.raw} score={c.score}  ({c.detail})")


if __name__ == "__main__":
    main()
