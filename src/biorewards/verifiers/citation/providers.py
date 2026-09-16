"""Resolution providers for the citation verifier.

A factual scientific claim is grounded against two external sources: a DOI
(does it resolve, and what does the paper actually say) and a database entity
(what is PDB 5EHF really). Both live behind a ``ResolverProvider`` so the reward
logic stays pure, offline, and testable, and so a real resolver drops in without
touching the scorer.

``MockResolver`` is a MOCK. It is a small, hardcoded, curated knowledge base for
reproducible offline testing and demos (no network). Its entries are drawn from a
real anti-drift incident: an agent claimed PDB 5EHF was a VEGFR2 complex when
5EHF is in fact a laccase. The mock catches exactly that class of fabrication.

``LiveResolver`` is the real provider. It hits Crossref for DOI metadata and RCSB
for PDB entities. Tests never touch it; it is guarded behind ``import requests``.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

# Crossref abstracts arrive as JATS-tagged XML (e.g. "<jats:p>...</jats:p>").
# Strip the tags to recover plain text; collapse the resulting whitespace.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_abstract(raw: str) -> str:
    if not raw:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()


@runtime_checkable
class ResolverProvider(Protocol):
    """Resolves DOIs and database entities to ground-truth descriptions.

    resolve_doi(doi) -> dict | None
      None if the DOI does not resolve. Otherwise a dict with keys:
        * title     the paper title
        * abstract  the paper abstract (free text, may contain quantities)
        * year      publication year (int)

    entity_identity(etype, eid) -> str | None
      A canonical description of a database entity (e.g. a PDB id maps to its
      molecule name). None if the entity is unknown to the resolver.
    """

    def resolve_doi(self, doi: str) -> dict | None:
        ...

    def entity_identity(self, etype: str, eid: str) -> str | None:
        ...


class MockResolver:
    """MOCK resolver. A curated in-memory knowledge base for offline testing.

    Every entry below is real ground truth, not fabricated. The DOI abstracts are
    trimmed stand-ins written to carry the numeric facts the tests assert on. The
    PDB entries include the 5EHF=laccase trap that motivated this verifier.

    Same query in, same answer out, on every machine. Swap in ``LiveResolver`` for
    live Crossref + RCSB resolution.
    """

    name = "mock_resolver"

    # Curated DOI knowledge base. Keys are lowercased DOIs. These use the 10.5555
    # test-DOI prefix on purpose: the abstracts are curated demo stand-ins, so they
    # must NOT be keyed to a real DOI (a verifier that catches misattribution should
    # not itself ship one). For live resolution use LiveResolver against Crossref.
    _DOIS: dict[str, dict] = {
        # Ramucirumab (IMC-1121B) Fab in complex with VEGFR2 domain 3. The abstract
        # carries the 3.37 nM affinity figure the tests match against (curated).
        "10.5555/demo.ramucirumab-vegfr2": {
            "title": "Structural basis of VEGFR2 recognition by the ramucirumab Fab",
            "abstract": (
                "The therapeutic antibody ramucirumab (IMC-1121B) targets vascular "
                "endothelial growth factor receptor 2 (VEGFR2). We report the crystal "
                "structure of the ramucirumab Fab bound to VEGFR2 domain 3 and show "
                "that the Fab binds VEGFR2 with an equilibrium dissociation constant "
                "of 3.37 nM, blocking ligand engagement at the receptor."
            ),
            "year": 2016,
        },
        # A resolvable paper that mentions VEGFR2 but carries a different number,
        # useful as a numeric-mismatch control.
        "10.1000/vegfr2-affinity-9p9": {
            "title": "Engineered VEGFR2 mini-binders and their affinities",
            "abstract": (
                "We designed de novo mini-binders against VEGFR2. The best design "
                "bound with a dissociation constant of 42.0 nM as measured by SPR."
            ),
            "year": 2024,
        },
    }

    # A DOI that is syntactically valid but does NOT resolve (tests the hard gate).
    _UNRESOLVABLE: set[str] = {"10.9999/does-not-exist"}

    # Curated entity knowledge base. Keys are (TYPE, ID) with uppercased id.
    _ENTITIES: dict[tuple[str, str], str] = {
        # THE TRAP. 5EHF is a laccase, not a VEGFR2 complex. An agent that claims
        # 5EHF is a VEGFR2 structure must be caught here.
        ("PDB", "5EHF"): "laccase",
        # Correct provenance for the ramucirumab / VEGFR2 co-crystal.
        ("PDB", "3S36"): "VEGFR2 domain 3 in complex with ramucirumab Fab",
    }

    def resolve_doi(self, doi: str) -> dict | None:
        key = doi.strip().lower()
        if key in self._UNRESOLVABLE:
            return None
        entry = self._DOIS.get(key)
        return dict(entry) if entry is not None else None

    def entity_identity(self, etype: str, eid: str) -> str | None:
        return self._ENTITIES.get((etype.strip().upper(), eid.strip().upper()))


class LiveResolver:
    """Real resolver. Crossref for DOI metadata, RCSB for PDB entities.

    Guarded behind ``import requests``. Tests use ``MockResolver`` and never
    reach the network. This is a real-provider sketch; a production deployment
    would add caching, retries, and rate-limit handling.
    """

    name = "live_resolver"

    # Identify ourselves to the public APIs (Crossref politeness pool, RCSB).
    _HEADERS = {"User-Agent": "biorewards/0.1 (mailto:rafael@molecule.to)"}

    def __init__(self, timeout: float = 10.0):
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LiveResolver needs the 'requests' package (pip install requests). "
                "For offline tests and demos use MockResolver instead."
            ) from exc
        self.timeout = timeout

    def resolve_doi(self, doi: str) -> dict | None:
        import requests

        url = f"https://api.crossref.org/works/{doi.strip()}"
        try:
            resp = requests.get(url, headers=self._HEADERS, timeout=self.timeout)
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        # message holds the work record; title is a list, abstract may be absent
        # and JATS-tagged when present, issued.date-parts nests the year first.
        msg = resp.json().get("message", {})
        title = " ".join(t for t in (msg.get("title") or []) if t)
        abstract = _clean_abstract(msg.get("abstract", "") or "")
        year = None
        parts = (msg.get("issued", {}) or {}).get("date-parts") or [[]]
        if parts and parts[0]:
            year = parts[0][0]
        return {"title": title, "abstract": abstract, "year": year}

    def entity_identity(self, etype: str, eid: str) -> str | None:
        import requests

        if etype.strip().upper() != "PDB":
            # Only PDB is wired up in this sketch. Extend for UniProt, ChEMBL, etc.
            return None
        url = f"https://data.rcsb.org/rest/v1/core/entry/{eid.strip().lower()}"
        try:
            resp = requests.get(url, headers=self._HEADERS, timeout=self.timeout)
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        # The structure title lives at struct.title.
        title = (resp.json().get("struct") or {}).get("title")
        return title or None
