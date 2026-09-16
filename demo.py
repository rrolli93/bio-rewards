"""One framework, four verifiers. Run: uv run python demo.py

Each verifier is an instance of the same R(x) = 1[gates] * sum_i w_i phi_i(g_i(x)).
A good candidate scores high, a bad one is caught. All offline, all deterministic.
"""

from biorewards.verifiers.protocol import PrimerPairReward, GuideReward
from biorewards.verifiers.binder import BinderReward
from biorewards.verifiers.metabolic import MetabolicReward
from biorewards.verifiers.citation import CitationReward


def show(title, rf, good, bad):
    g = rf.evaluate(good)
    b = rf.evaluate(bad)
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    print(f"  GOOD  reward={g.reward}  status={g.status}")
    print(f"  BAD   reward={b.reward}  status={b.status}"
          + (f"  [{b.hard_gate_failures[0]}]" if b.hard_gate_failures else ""))


show("PROTOCOL  (PCR primer pair)  Tm/GC/clamp/dimer, nearest-neighbor",
     PrimerPairReward(),
     {"fwd": "ACCACAGTCCATGCCATCAC", "rev": "TCCACCACCCTGTTGCTGTA"},
     {"fwd": "GGGGGGGGGGGGGGGGGGGG", "rev": "ATATATATATATATATATAT"})

show("PROTOCOL  (CRISPR guide)  PAM + terminator + composition",
     GuideReward(),
     {"guide": "GACGCATCGTACGATCGTAC", "pam": "AGG"},
     {"guide": "GACGCATCGTACGATCGTAC", "pam": "TAA"})

show("BINDER  (peptide design)  developability + fold confidence",
     BinderReward(),
     {"sequence": "SEDKEAWNTGCKQFVDSNGRTHCLDPEKAQY"},
     {"sequence": "WWLLIIFFVVWWLLIIFFVVKRKRKRKRKRC"})

show("METABOLIC  (knockout strategy)  flux balance analysis",
     MetabolicReward(),
     {"knockouts": ["EX_byproduct"], "objective": "EX_target"},
     {"knockouts": ["BIOMASS"], "objective": "EX_target"})

show("CITATION  (claim grounding)  DOI + number + provenance",
     CitationReward(),
     {"claim": "Ramucirumab Fab binds VEGFR2 at 3.37 nM (PDB 3S36).",
      "doi": "10.5555/demo.ramucirumab-vegfr2",
      "value": {"number": 3.37, "unit": "nM", "tolerance": 0.1},
      "entity": {"type": "PDB", "id": "3S36", "expected": "VEGFR2"}},
     {"claim": "PDB 5EHF is the VEGFR2 complex bound at 3.37 nM.",
      "doi": "10.5555/demo.ramucirumab-vegfr2",
      "value": {"number": 3.37, "unit": "nM", "tolerance": 0.1},
      "entity": {"type": "PDB", "id": "5EHF", "expected": "VEGFR2"}})

print()
