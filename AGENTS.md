# AGENTS.md — bio-rewards

**Read this first. It gives you (a Codex / GPT-Rosalind / any coding agent) the full
context of this project so you can continue with zero prior knowledge.**

Repo: https://github.com/rrolli93/bio-rewards . Local: `~/Developer/bio-rewards`.
Author: Rafa Rolli (Science Lead, PeptAI / Molecule). Built for the **NVIDIA x OpenAI
London AIxBio hackathon, 18-20 Sep 2026** (WestWorks, White City).

---

## 1. The one idea

Reasoning models got trustworthy in math and code because correctness is **cheap to
check**: a proof verifier, a unit test, a compiler. That loop is RLVR (reinforcement
learning with verifiable rewards). Biology lacks those verifiers, so bio-agents
hallucinate with confidence. **bio-rewards is a library of deterministic verifiers for
biology.** Hallucination is a feature when generating hypotheses and a defect in any
workflow that needs precision. This library is the precision half.

The hackathon framing (do not drift from this): **the deterministic reward layer is the
star.** An OpenAI agent (GPT-Rosalind / Codex) proposes candidates, NVIDIA GPUs (Brev /
BioNeMo / Boltz) do heavy compute, and our reward layer verifies and selects, so the
agent converges on real answers instead of fabricating. Framing track = **Orchestration**
(single prize track; the website "tracks" are only suggestions). Do NOT center folding /
binders alone (crowded lane); center the reward-function breadth across domains.

## 2. The contract (one skeleton, every verifier)

```
R(x) = 1[hard gates pass] * sum_i  w_i * phi_i( g_i(x) )
```

- `x` = the candidate the agent proposes. It only ever passes `x`. It never touches the
  scoring code. Sealed boundary = the safety story.
- hard gates = binary validity predicates; any failure => reward exactly 0.
- `g_i(x)` = a deterministic measurement (a Tm, a docking energy, a metabolic flux, ipSAE).
- `phi_i` = a calibrated membership function mapping the measurement to [0,1]. Trapezoids
  encode biological sweet spots (too little bad AND too much bad).
- Everything is Python `Decimal` with fixed rounding and a hashed config, so the same
  input yields the same bytes and results cache and reproduce.

Five trust properties: deterministic, calibrated (anchors from reference panels before
scoring novel candidates), hard gates separate from graded score, honest applicability
domain (each result says whether it is in the verifier's competence), Goodhart-resistant.

**Provider pattern:** expensive/external steps (folding, docking, FBA, DOI lookup) are
injected providers, each with a Mock fallback, so the reward logic is pure and every demo
runs offline (no GPU, no network). Real providers plug in without touching the scorer.

## 3. Code map

```
src/biorewards/
  core.py         RewardFunction (subclass this), Component, Applicability, RewardResult
  membership.py   trapezoid / rising / falling / anchored  (the phi_i)
  optimize.py     optimize(reward_fn, seed, mutate, ...) -> OptimizeResult with .trajectory
                  (seeded evolutionary hill-climb; best-so-far is monotone => converges)
  mutators.py     dna_primer_pair_mutator, protein_mutator  (domain mutation operators)
  viz.py          plot_convergence(result, path)  (one-call convergence plot, any domain)
  verifiers/
    protocol/     PCR primers (SantaLucia nearest-neighbor Tm) + CRISPR guides (PAM/etc)
    binder/       protein/peptide binder: developability + fold confidence
    metabolic/    metabolic-engineering knockouts scored by flux balance analysis
    citation/     factual-claim grounding: DOI resolves + number-in-source + entity check
    smallmolecule/ drug-likeness + synthetic accessibility + docking  (RDKit)
tests/            pytest; offline tests always run, heavy-dep tests importorskip
deck/             pitch deck (index.html), explainer pages (how-it-works.html, explain.html),
                  figures/ (landscape, convergence, 4-domain grid, gif)
scripts/          demo + visualization + live-fold scripts
demo.py           one-command showpiece across the offline verifiers
```

Each verifier is a `RewardFunction` subclass implementing `hard_gates`, `score_components`,
`applicability`, `config`, `identity`. Copy `verifiers/protocol/reward.py` as the template.

## 4. The five verifiers (what each measures; real vs mock)

- **protocol** (fully real, offline). Primer melting temperature via SantaLucia 1998
  nearest-neighbor thermodynamics, GC, 3' clamp, self-complementarity; CRISPR PAM +
  terminator + composition. Validated on real GAPDH primers (0.935) vs pathological (0.000).
- **binder** (real via boltz-api; mock offline). Developability from sequence (GRAVY,
  net charge, aromatic fraction, cysteine parity, Kex2 sites, hydrophobic patch) + fold
  confidence: **pLDDT, ipTM, ipSAE, pae_interaction**. ipSAE and pae_interaction are
  computed BY US from the PAE matrix (`binder/ipsae.py`, Dunbar & Sternberg 2024), so the
  folder is swappable. Target-aware: co-folds the complex; interface terms drop when no
  target. If a folder returns no PAE (LiteFold), it scores ipTM + pLDDT only.
- **metabolic** (real via cobra; toy mock offline). Flux balance analysis (a linear
  program): product yield at viable growth + growth-coupling. `CobraFBAProvider` reproduces
  the E. coli core textbook wild-type growth 0.8739/h.
- **citation** (real via live Crossref+RCSB; mock offline). DOI resolves + asserted number
  is in the source + database entity provenance is correct. Encodes the real anti-drift
  case (PDB 5EHF is a laccase, not VEGFR2) as a hard-gated fabrication.
- **smallmolecule** (RDKit real; mock docking). Drug-likeness (MW, cLogP, TPSA, HBD, HBA,
  rotatable bonds) + synthetic accessibility + a docking/binding term (Mock offline; real =
  Rowan/Vina as in `~/peptai/molecule-optimization-sandbox`). PAINS as a hard gate.

## 5. Connectors / keys (state as of 2026-09-17)

Secrets live in `~/.config/peptai/.env` (chmod 600, symlinked from repo roots). Never
commit or echo raw keys.

- **Folding — boltz-api (WIRED, WORKING).** `BoltzApiFoldProvider` shells the authenticated
  `boltz-api` CLI. Live compute API schema: bare protein entities (no per-entity `msa` /
  `modifications`; the server generates MSAs). Result: `metrics.json` gives `iptm` and
  `complex_plddt` (0-1, rescale x100); `sample_N_pae.npz` gives the PAE. Auth: export
  `BOLTZ_API_KEY` from `BOLTZ_API_KEY_LIVE` (OAuth session may be expired; use the key).
  Cost ~ $0.05 for a peptide+GPCR fold, ~minutes. Each fold caches as its run dir.
- **Folding — LiteFold (WIRED).** `LiteFoldFoldProvider`, hosted agents API
  `https://agentsapi.litefold.ai` (NOT `api.litefold.ai`), Bearer `LITEFOLD_API_KEY`.
  Returns ipTM + pLDDT but NO PAE, so ipSAE/pae_interaction are skipped for this folder.
- **Folding — subseq.bio (NOT wired).** Base `https://subseq.bio/api/v1`, Bearer `sk-ss-`
  key, its Chai output DOES include PAE. Not wired because its co-fold submit/result
  contract needs one live smoke-test and its cluster can be offline. Good future addition.
- **Metabolic — cobra.** `uv run --with cobra ...`; `CobraFBAProvider` loads the bundled
  `textbook` E. coli core model.
- **Citation — live.** `LiveResolver` hits Crossref (`api.crossref.org/works/{doi}`) and
  RCSB (`data.rcsb.org/rest/v1/core/entry/{id}`). Needs `requests`.
- **Small molecule — RDKit.** `uv run --with rdkit ...`. Real docking = Rowan (see the
  molecule sandbox), currently a documented stub here.

## 6. Live results already produced (real, reproducible)

- **Primer design:** a nonsense seed (reward 0) optimized to a synthesizable pair with
  matched 59/60 C melting temps, reward 1.000. (`scripts/live_loop.py`)
- **KISS1R C1-01 scaffold fold** (binder `YNWNSFGLRF` = KP-10 backbone of the Eurofins
  candidate `{D-Tyr}NWNSFGL{Arg(Me)}F-NH2`; ncAA mods NOT represented) vs human KISS1R
  (Q969F8): reward **0.858**, ipTM 0.912, ipSAE 0.643, pae_interaction 7.83 A (below the
  10 A good-binder threshold), pLDDT 72. All three interface metrics agree given a real
  receptor MSA. (`scripts/kiss1r_c101_fold.py`)
- **Metabolic (real E. coli):** searching single knockouts to overproduce acetate, the
  reward picks `CYTBD` (cytochrome oxidase) knockout, which growth-couples acetate
  (coupling 0 -> 8.5), a textbook-correct strain-engineering result.

## 7. How to run

```bash
# offline showpiece (no GPU, no network, no keys):
uv run python demo.py

# full test suite (offline; heavy-dep tests skip):
uv run --with pytest --with numpy python -m pytest -q
# with the heavy deps so those tests run:
uv run --with pytest --with numpy --with cobra --with requests --with rdkit python -m pytest -q

# a live search you can watch converge (any domain):
uv run --with matplotlib python scripts/live_loop.py

# a real fold + score (needs BOLTZ_API_KEY exported):
export BOLTZ_API_KEY=$(grep '^BOLTZ_API_KEY_LIVE=' ~/.config/peptai/.env | cut -d= -f2-)
uv run --with numpy python scripts/kiss1r_c101_fold.py

# convergence figures + the four-domain grid:
uv run --with matplotlib --with pillow --with numpy python scripts/visualize_all.py
```

## 8. What to build next (the hackathon plan)

The differentiated + SHINY deliverable is a **live discovery cockpit**: an OpenAI agent
(GPT-Rosalind) proposes candidates across domains, the deterministic reward drives the
loop, and the UI renders the payoff live: a reward gauge/curve climbing, the current best
candidate rendered (3D complex from the Boltz CIF via Mol*/NGL; a small molecule drawn
from SMILES; a primer Tm dial), and a "caught red-handed" feed where hallucinations get
rejected on screen with the deterministic reason (e.g. "PDB 5EHF is a laccase, not
VEGFR2"). Judging criteria (5, equal): scientific relevance, effective+central use of
NVIDIA and OpenAI, execution, originality, presentation/reproducibility.

Concrete integration TODO for the OpenAI/NVIDIA platforms:
1. Swap boltz-api for **Boltz/BioNeMo on NVIDIA Brev** (GPU) as a `FoldProvider`.
2. Wire **GPT-Rosalind / Codex** as the generator: it emits candidates (SMILES, sequences,
   knockout sets), `RewardFunction.evaluate` scores them, feed the reward + component
   breakdown back as the agent's signal; loop until convergence. (See `optimize.py` for the
   deterministic-mutator version of this loop to mirror.)
3. Build the cockpit UI (self-contained web page; render structures with Mol*/NGL from CIF,
   molecules with a SMILES drawer, live reward curve).
4. Add a SMILES mutator (RDKit) if you want a deterministic small-molecule search too.

## 9. Conventions (match these exactly)

- Voice: **no em-dashes or en-dashes** anywhere (code, comments, docs, output). Periods and
  parentheses. Sparse, meaningful comments.
- Determinism: all scores `Decimal`; providers cache; no wall-clock or RNG in scoring.
- New verifier = a `RewardFunction` subclass in its own `verifiers/<name>/` dir + a
  `tests/test_<name>.py`; guard heavy-dep tests with `pytest.importorskip`.
- Never weaken existing tests. The default `uv run --with pytest python -m pytest -q` must
  stay green with no heavy deps installed.
- Honesty over shine in the code: if a term is mocked, say so in `applicability`. Real vs
  mock must always be legible.

## 10. Related context (outside this repo)

- `~/peptai/molecule-optimization-sandbox` — the ancestor small-molecule reward engine
  (drug_likeness + purchasability + Rowan docking + MCTS). The `smallmolecule` verifier
  here is its port into the bio-rewards contract. `HANDOFF.md` there has the CYP51 arc.
- PeptAI vault (`~/peptai`, an Obsidian vault, not this repo) — programme context for
  KISS1R/OX2R (the C1-01 candidate), the agonism predictor, etc. Not needed to work on
  bio-rewards, but it is where the science candidates come from.
