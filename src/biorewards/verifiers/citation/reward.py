"""Reward function for grounding a factual scientific claim.

CitationReward -- turn anti-hallucination discipline into a deterministic reward.

LLM science agents fabricate citations with confidence: DOIs that do not resolve,
numbers not present in the source, and wrong entity provenance (claiming PDB 5EHF
is a VEGFR2 complex when 5EHF is a laccase). This verifier catches all three.

DOI resolution, the numeric match, and the entity-type check are HARD
deterministic. Only the final claim-vs-abstract entailment is model-shaped, and
here it is a deterministic lexical-overlap stand-in flagged as such in
applicability. A real deployment swaps in an NLI entailment model.
"""

from __future__ import annotations

import re
from decimal import Decimal

from biorewards.core import Applicability, Component, RewardFunction
from biorewards.membership import rising

from .providers import MockResolver, ResolverProvider

# A basic DOI shape. Not a full grammar, just enough to reject obvious garbage.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

# Pull signed decimal numbers out of source text (e.g. "3.37", "42.0", "-10").
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Content-word tokenizer for the lexical-overlap entailment proxy.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Common English function words dropped before computing overlap so the proxy
# tracks content, not grammar.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that
    the to was were with we our this these those which at than then""".split()
)


class CitationReward(RewardFunction):
    """Score a factual claim for grounding against its cited source.

    Expects x = {
        "claim": "<free-text scientific claim>",
        "doi": "<doi string>",
        optional "value": {"number": 3.37, "unit": "nM", "tolerance": 0.1},
        optional "entity": {"type": "PDB", "id": "5EHF", "expected": "VEGFR2"},
    }
    """

    name = "citation"
    version = "0.1.0"

    def __init__(self, resolver: ResolverProvider | None = None,
                 support_edges=(0.05, 0.4)):
        self.resolver = resolver if resolver is not None else MockResolver()
        self.support_edges = support_edges

    def config(self) -> dict:
        return {
            "resolver": getattr(self.resolver, "name", type(self.resolver).__name__),
            "support_edges": self.support_edges,
        }

    def identity(self, x) -> str:
        import json

        return json.dumps(
            {
                "claim": x.get("claim", ""),
                "doi": x.get("doi", ""),
                "value": x.get("value"),
                "entity": x.get("entity"),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def hard_gates(self, x) -> list[str]:
        fails = []
        doi = (x.get("doi") or "").strip()
        if not doi:
            fails.append("doi: missing")
            return fails
        if not _DOI_RE.match(doi):
            fails.append(f"doi: '{doi}' is not a syntactically valid DOI")
            return fails
        # A DOI that does not resolve cannot ground anything. Full stop.
        if self.resolver.resolve_doi(doi) is None:
            fails.append("doi does not resolve")
            return fails
        # A KNOWN-and-contradicted entity provenance is a fabrication, not a soft
        # deduction. If the database says 5EHF is a laccase and the claim says it
        # is VEGFR2, that is a factual error and the whole claim fails. An UNKNOWN
        # entity (truth is None) stays unverifiable and is simply not scored.
        entity = x.get("entity")
        if entity is not None:
            truth = self.resolver.entity_identity(entity["type"], entity["id"])
            if truth is not None and not _token_match(entity["expected"], truth):
                fails.append(
                    f"entity provenance contradicted: {entity['type']} "
                    f"{entity['id']} is '{truth}', claimed '{entity['expected']}'"
                )
        return fails

    def score_components(self, x) -> list[Component]:
        doi = x["doi"].strip()
        meta = self.resolver.resolve_doi(doi)  # gate guarantees this is not None
        source_text = f"{meta.get('title', '')} {meta.get('abstract', '')}"

        components: list[Component] = []

        value = x.get("value")
        if value is not None:
            hit = _number_in_text(
                source_text,
                float(value["number"]),
                float(value.get("tolerance", 0.0)),
            )
            components.append(
                Component(
                    "number_in_source",
                    1.0 if hit else 0.0,
                    Decimal("1") if hit else Decimal("0"),
                    Decimal("0.4"),
                    f"{value['number']}{value.get('unit', '')} "
                    f"{'found' if hit else 'absent'} in source",
                )
            )

        entity = x.get("entity")
        if entity is not None:
            truth = self.resolver.entity_identity(entity["type"], entity["id"])
            # A contradicted entity was already hard-gated. Here truth either
            # matches (credit it) or is unknown (omit, unverifiable).
            if truth is not None:
                components.append(
                    Component(
                        "entity_type_correct",
                        1.0,
                        Decimal("1"),
                        Decimal("0.4"),
                        f"{entity['type']} {entity['id']} is '{truth}', "
                        f"matches claimed '{entity['expected']}'",
                    )
                )

        # Deterministic lexical-overlap stand-in for NLI entailment.
        overlap = _jaccard_overlap(x.get("claim", ""), source_text)
        a, b = self.support_edges
        components.append(
            Component(
                "claim_support",
                round(overlap, 4),
                rising(overlap, a, b),
                Decimal("0.2"),
                f"claim/source content overlap {overlap:.2f} (lexical proxy)",
            )
        )
        return components

    def applicability(self, x) -> Applicability:
        live = getattr(self.resolver, "name", "") == "live_resolver"
        source = (
            "LiveResolver hits Crossref + RCSB."
            if live
            else "With MockResolver the knowledge base is curated for the demo."
        )
        return Applicability(
            in_domain=True,
            reason=(
                "DOI resolution, numeric match, and entity-type check are "
                "hard-deterministic. claim_support here is a lexical-overlap "
                "proxy; a real deployment should swap in an NLI entailment "
                f"model. {source}"
            ),
        )


def _number_in_text(text: str, target: float, tolerance: float) -> bool:
    """True if any number parsed from ``text`` is within tolerance of target."""
    for tok in _NUM_RE.findall(text):
        try:
            val = float(tok)
        except ValueError:
            continue
        if abs(val - target) <= tolerance:
            return True
    return False


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS}


def _token_match(expected: str, truth: str) -> bool:
    """Case-insensitive token/substring match of expected against a truth string.

    True when the expected string is a substring of the truth, or when every
    content token of the expected appears in the truth's token set. This treats
    'VEGFR2' vs 'laccase' as a contradiction (0.0) and 'VEGFR2' vs 'VEGFR2
    domain 3 ... ramucirumab' as a match (1.0).
    """
    exp = expected.strip().lower()
    tru = truth.strip().lower()
    if not exp:
        return False
    if exp in tru:
        return True
    exp_tokens = _content_tokens(expected)
    if not exp_tokens:
        return False
    return exp_tokens <= _content_tokens(truth)


def _jaccard_overlap(claim: str, source: str) -> float:
    """Jaccard overlap of content-word token sets. Deterministic entailment proxy."""
    a = _content_tokens(claim)
    b = _content_tokens(source)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
