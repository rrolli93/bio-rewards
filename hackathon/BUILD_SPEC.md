# Protein Design Harness — Build Spec

**Team:** Carlos Sainz (captain), Rafael Rolli, Demilade Odetara
**Event:** NVIDIA x OpenAI London AIxBio Hack, 18-20 Sep 2026
**Repo (Rafa's reusable scoring + this spec + benchmark):** github.com/rrolli93/bio-rewards
**Base tool to fork:** https://github.com/aurekaresearch/OpenDDE-Harness (Apache-2.0)

## One line

An agent designs nanobodies (VHH), NVIDIA GPUs fold the complex, and a reward that
**combines fold-confidence metrics (ipSAE) with PyRosetta physics (interface dG)**
tells real binders from non-binders far better than confidence alone, closing the
design loop.

## Why (the science, cite these on the slides)

- Fold confidence alone (ipTM, pLDDT) is a weak binder/non-binder discriminator.
- bioRxiv 2025.08.14.670059 (3,766 tested binders, 15 targets): **ipSAE** beats older
  confidence metrics (1.4x average precision over ipAE); adding **Rosetta dG/dSASA**
  and shape complementarity improves further. Confidence + physics is the win.
- PMC12172308 (NanoBinder): for nanobodies specifically, Rosetta interface features
  (**dG_cross/dSASA**, interface H-bonds, buried unsatisfied H-bonds, fa_atr) classify
  binders at ~92% accuracy. Physics carries real signal.
- OpenDDE-Harness gives the agentic design loop but feeds back **confidence-only**.
  Our contribution is the missing physics-augmented reward.

## Architecture (the loop)

```
  OpenAI agent (GPT / Codex / Rosalind)        <- proposes / refines VHH CDR sequences
        |                                          (keep framework fixed, design CDRs)
        v
  [optional] NVIDIA ProteinMPNN NIM             <- CDR sequence design on the backbone
        |
        v
  NVIDIA folding NIM on Brev                    <- Boltz-2 (start here) or AF2-multimer
   (VHH + antigen complex)                         + msa-search; outputs structure + PAE
        |
        v
  REWARD  =  combine(                           <- Rafa's module
     confidence:  ipSAE, ipTM, pLDDT, pae_interaction   (from PAE)
     physics:     dG, dG/dSASA, hbonds_int, dSASA, shape complementarity (PyRosetta)
  )
        |
        v
  feedback score  -> agent iterates            <- better reward = better designs
```

NVIDIA is central (Brev GPUs + BioNeMo folding NIM, optionally ProteinMPNN). OpenAI is
central (the designer agent). Neither is incidental.

## Division of labor (adjust to strengths)

- **Carlos (protein design + physics):** PyRosetta interface module. Input a predicted
  VHH-antigen complex (PDB/CIF), output the physics features via `InterfaceAnalyzerMover`:
  dG_separated, dG/dSASA (dG_cross/dSASA), hbonds_int, delta_unsat_hbonds, packstat, and
  shape complementarity (`sc`). One function: `physics_features(pdb_path, binder_chain,
  target_chain) -> dict`. Get PyRosetta licensed + installed Saturday morning.
- **Rafa (scoring + proof):** the confidence metrics (reuse `biorewards/verifiers/binder/
  ipsae.py` for ipSAE + pae_interaction from the PAE) and the **combined reward** that
  fuses confidence + physics. Owns `hackathon/benchmark.py` (the winning number) and the
  combined-scorer used as the loop's feedback. Also the reproducible repo + tests.
- **Demi (harness + NVIDIA wiring + demo):** stand up OpenDDE-Harness, wire the Brev
  folding NIM (Boltz-2) so it returns structure + PAE, connect the reward as the harness
  feedback signal, and own the demo recording + Google Slides. (Flex with Carlos/Rafa.)

## Milestones (timeboxed)

**Saturday AM**
- OpenDDE-Harness running (`uv tool install --python 3.12 opendde-harness`, `ddeharness
  onboard`, `ddeharness doctor`). Agent designs one VHH end to end on its own folding.
- PyRosetta licensed + installed on the Brev box. Boltz-2 NIM reachable on Brev.
- Decide: fold via BioNeMo NIM (preferred, we own the PAE) vs harness folding.

**Saturday PM**
- Carlos: `physics_features()` returns real numbers on a test complex.
- Rafa: confidence metrics from PAE + a combined scorer; `benchmark.py` runs on a real
  or curated labeled set.
- Demi: reward wired into the harness loop; one iteration visibly improves the score.

**Sunday AM**
- The benchmark number: confidence-only AUROC vs confidence+physics AUROC on a held-out
  labeled set. This is the result that wins.
- One clean end-to-end demo run (recorded as backup).

**Sunday 1:30pm** presentation prep. **3pm** final presentations (5 min).

## The result that wins

A single chart: **AUROC (and average precision) of confidence-only vs physics-only vs
combined** on a labeled binder/non-binder set. If combined > confidence-only, that one
number satisfies three judging criteria at once (scientific relevance, execution,
reproducible evidence). See `hackathon/benchmark.py`. Get a real labeled set if possible
(the bioRxiv/NanoBinder supplements, SAbDab, or the harness's own examples); the script
ships a synthetic generator so the pipeline is provable before real data lands.

## Submission checklist (Sunday)

- [ ] Google Slides using the org template (deterministic story: problem, loop, the
      NVIDIA+OpenAI stack, the benchmark number, demo).
- [ ] Single GitHub repo (the OpenDDE-Harness fork + our physics module + reward +
      benchmark), clean README with run instructions, so a stranger can reproduce.
- [ ] Functioning demo (recorded backup so nothing breaks live).

## Early risks to kill in the first hour

- **PyRosetta license + install** (free for academics; Carlos/DTU qualifies) can be
  fiddly. Do it first.
- **Does the folding step expose the PAE?** ipSAE needs it. Boltz-2 NIM returns it; if
  using the harness's own folding, confirm PAE access or fold with Boltz-2 on the side.
- **Brev NIM access** only unlocks at event start. Confirm the Boltz-2 NIM runs on your
  Brev instance before building on it.
