"""Fold providers for the binder verifier.

Scoring a binder means scoring the binder-target COMPLEX, not the binder alone.
So a provider co-folds the complex when given a target and returns the raw fold
outputs the reward reads: a per-chain pLDDT, an interface pTM (ipTM), and the
full PAE matrix. The interface numbers we actually score (ipSAE,
pae_interaction) are computed by us from that PAE in ``ipsae.py``, so the folder
is swappable without touching the scorer.

``MockFoldProvider`` is a MOCK. Only the PAE INPUT it emits is fabricated (from a
hash of the sequences); the ipSAE math run on top of it in ``ipsae.py`` is the
real, published computation. It exists to make the reward reproducible and
demonstrable with no GPU and no network. Swap ``BoltzFoldProvider`` (once wired)
for structural confidence that means something.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class FoldProvider(Protocol):
    """Co-folds a binder (optionally with a target) and returns raw fold outputs.

    fold(binder, target) -> dict with keys:
      * plddt      mean pLDDT over the binder chain, [0, 100], higher better
      * iptm       interface pTM, [0, 1], higher better; None when target is None
      * pae        (N x N) nested list, N = n_binder + n_target, ordered
                   [binder residues, then target residues]; may be None if the
                   folder does not expose a PAE matrix (then ipSAE and
                   pae_interaction are skipped and only ipTM + pLDDT are scored)
      * n_binder   int
      * n_target   int (0 when target is None)
    """

    def fold(self, binder: str, target: str | None) -> dict:
        ...


class MockFoldProvider:
    """MOCK fold provider. Deterministic, offline, scientifically meaningless.

    A quality scalar q in [0, 1] is derived from sha256(binder + "/" + target).
    From q it builds:
      * a PAE matrix with small intra-chain entries (~2-6 A) and cross-chain
        entries around (30 - q * 27), each with a small deterministic per-cell
        jitter, clamped to [0.5, 30]
      * iptm = round(0.2 + q * 0.7, 3) when a target is present, else None
      * plddt = round(50 + q * 45, 2)

    A high-q binder therefore shows low interface PAE, high ipTM, and high pLDDT
    together, and (through the real ipsae.py math) a high ipSAE. Only the PAE
    INPUT is fake; the ipSAE computation layered on it is the genuine one.
    """

    name = "mock_fold"

    def _quality(self, binder: str, target: str) -> float:
        digest = hashlib.sha256(f"{binder}/{target}".encode()).digest()
        return int.from_bytes(digest[0:4], "big") / 0xFFFFFFFF

    def _jitter(self, seed: bytes, i: int, j: int, spread: float) -> float:
        """Deterministic per-cell wobble in [-spread, spread] from a hash byte."""
        b = hashlib.sha256(seed + i.to_bytes(4, "big") + j.to_bytes(4, "big")).digest()
        return (b[0] / 255.0 * 2.0 - 1.0) * spread

    def fold(self, binder: str, target: str | None) -> dict:
        binder = binder.upper()
        tgt = (target or "").upper()
        n_binder = len(binder)
        n_target = len(tgt)
        n = n_binder + n_target

        q = self._quality(binder, tgt)
        seed = hashlib.sha256(f"pae/{binder}/{tgt}".encode()).digest()
        cross_center = 30.0 - q * 27.0

        pae = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                same_chain = (i < n_binder) == (j < n_binder)
                if same_chain:
                    base = 4.0 + self._jitter(seed, i, j, 2.0)   # ~2-6 A
                    pae[i][j] = min(6.0, max(2.0, base))
                else:
                    val = cross_center + self._jitter(seed, i, j, 1.5)
                    pae[i][j] = min(30.0, max(0.5, val))

        iptm = round(0.2 + q * 0.7, 3) if n_target > 0 else None
        plddt = round(50.0 + q * 45.0, 2)

        return {
            "plddt": plddt,
            "iptm": iptm,
            "pae": pae,
            "n_binder": n_binder,
            "n_target": n_target,
        }


class BoltzApiFoldProvider:
    """Real co-folding via the authenticated ``boltz-api`` CLI (Boltz-2).

    Submits the binder as chain A and the target (when present) as chain B, waits
    for the GPU job, and parses the downloaded results into the FoldProvider
    contract: the best sample's ipTM, its complex pLDDT (rescaled to 0-100), and
    the full PAE matrix. ipSAE and pae_interaction are then computed by us in
    ``ipsae.py``, so the folder stays swappable.

    A real fold is a GPU job of minutes and it costs credits, so each fold is
    cached as its run directory: if the parsed results already exist for a
    (binder, target, model, num_samples) key, no new job is submitted. This makes
    the provider a shortlist-scorer (fold the best few designs), not an in-loop
    operation. Use MockFoldProvider inside a fast search, this to confirm winners.
    """

    name = "boltz_api_fold"

    def __init__(self, model="boltz-2.1", num_samples=1, msa="empty",
                 root_dir=None, cli="boltz-api", timeout=2400):
        import os
        self.model = model
        self.num_samples = num_samples
        self.msa = msa  # "empty" skips MSA generation (fast); use "auto" for a real MSA
        self.cli = cli
        self.timeout = timeout
        self.root_dir = root_dir or os.path.expanduser("~/.cache/biorewards/boltz")

    def _key(self, binder: str, target: str) -> str:
        payload = f"{binder}|{target}|{self.model}|{self.num_samples}|{self.msa}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _input(self, binder: str, target: str) -> dict:
        # The compute API generates MSAs server-side and rejects per-entity msa or
        # modifications properties, so entities are bare (chain, type, sequence).
        entities = [{"chain_ids": ["A"], "type": "protein", "value": binder}]
        spec = {"entities": entities, "num_samples": self.num_samples}
        if target:
            entities.append({"chain_ids": ["B"], "type": "protein", "value": target})
            spec["binding"] = {"binder_chain_ids": ["A"], "type": "protein_protein_binding"}
        return spec

    def fold(self, binder: str, target: str | None) -> dict:
        import json
        import os
        import subprocess

        binder = binder.upper()
        tgt = (target or "").upper()
        run_dir = os.path.join(self.root_dir, self._key(binder, tgt))
        pred = os.path.join(run_dir, "extracted", "prediction")

        # cache hit: results already downloaded for this exact request
        if not os.path.exists(os.path.join(pred, "metrics.json")):
            os.makedirs(run_dir, exist_ok=True)
            spec_path = os.path.join(run_dir, "input.json")
            with open(spec_path, "w") as f:
                json.dump(self._input(binder, tgt), f)
            env = dict(os.environ)
            env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
            subprocess.run(
                [self.cli, "predictions:structure-and-binding", "run",
                 "--input", f"@json://{spec_path}", "--model", self.model,
                 "--run-dir", run_dir],
                check=True, timeout=self.timeout, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        return self.parse_run_dir(run_dir, len(binder), len(tgt))

    @staticmethod
    def parse_run_dir(run_dir: str, n_binder: int, n_target: int) -> dict:
        """Parse a downloaded boltz-api run directory into the fold contract.

        Picks the sample with the highest ipTM, rescales complex pLDDT to 0-100,
        and loads that sample's PAE matrix. Works on any prior run directory,
        which is how this pipeline is validated without spending a fold.
        """
        import json
        import os
        import tarfile

        pred = os.path.join(run_dir, "extracted", "prediction")
        if not os.path.exists(os.path.join(pred, "metrics.json")):
            archive = os.path.join(run_dir, "outputs", "archive.tar.gz")
            if os.path.exists(archive):
                with tarfile.open(archive) as t:
                    t.extractall(os.path.join(run_dir, "extracted"))

        with open(os.path.join(pred, "metrics.json")) as f:
            metrics = json.load(f)
        samples = metrics.get("all_sample_results") or [metrics.get("best_sample", {})]
        idx = max(range(len(samples)),
                  key=lambda i: samples[i]["metrics"].get("iptm", 0.0))
        m = samples[idx]["metrics"]

        import numpy as np
        pae_path = os.path.join(pred, f"sample_{idx}_pae.npz")
        if not os.path.exists(pae_path):  # fall back to whatever pae file exists
            paes = [p for p in os.listdir(pred) if p.endswith("_pae.npz")]
            pae_path = os.path.join(pred, sorted(paes)[0])
        pae = np.load(pae_path)["pae"].tolist()

        return {
            "plddt": round(float(m.get("complex_plddt", 0.0)) * 100.0, 2),
            "iptm": round(float(m.get("iptm", 0.0)), 4),
            "pae": pae,
            "n_binder": n_binder,
            "n_target": n_target,
        }


# Back-compat alias.
BoltzFoldProvider = BoltzApiFoldProvider


def _load_env_key(name: str) -> str | None:
    """Read a key from the environment, falling back to ~/.config/peptai/.env."""
    import os

    if os.environ.get(name):
        return os.environ[name]
    path = os.path.expanduser("~/.config/peptai/.env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class LiteFoldFoldProvider:
    """Co-folding via the hosted LiteFold agents API (Boltz-2).

    LiteFold returns ipTM and complex pLDDT but NOT a PAE matrix, so this provider
    sets pae=None and the reward scores ipTM + pLDDT only (ipSAE and
    pae_interaction are skipped). That is the honest tradeoff of this folder versus
    boltz-api, which does expose the PAE. Use whichever the task needs, or both.

    Flow: init project, upload a 2-record FASTA (binder chain A, target chain B),
    submit Boltz-2, poll, read confidence_json. Each fold caches its parsed result.
    Needs LITEFOLD_API_KEY (read from env or ~/.config/peptai/.env).
    """

    name = "litefold_fold"
    BASE = "https://agentsapi.litefold.ai"

    def __init__(self, api_key=None, base_url=None, cache_dir=None,
                 poll_interval=10.0, timeout=2400):
        import os
        self.api_key = api_key or _load_env_key("LITEFOLD_API_KEY")
        self.base = (base_url or _load_env_key("LITEFOLD_BASE_URL") or self.BASE).rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/biorewards/litefold")

    def _key(self, binder: str, target: str) -> str:
        return hashlib.sha256(f"{binder}|{target}".encode()).hexdigest()[:16]

    def _headers(self):
        if not self.api_key:
            raise RuntimeError(
                "LiteFoldFoldProvider needs LITEFOLD_API_KEY (env or ~/.config/peptai/.env)."
            )
        return {"Authorization": f"Bearer {self.api_key}"}

    def fold(self, binder: str, target: str | None) -> dict:
        import json
        import os
        import time

        try:
            import requests
        except ImportError as exc:
            raise ImportError("LiteFoldFoldProvider needs 'requests'.") from exc

        binder = binder.upper()
        tgt = (target or "").upper()
        key = self._key(binder, tgt)
        os.makedirs(self.cache_dir, exist_ok=True)
        cache = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)

        job = f"biorewards-{key}"
        fasta = ">A\n" + binder + ("\n>B\n" + tgt if tgt else "") + "\n"
        h = self._headers()

        requests.post(f"{self.base}/jobs/init", json={"job_name": job},
                      headers=h, timeout=60)
        requests.post(f"{self.base}/fs/upload", headers=h, timeout=120,
                      files={"file": ("seqs.fasta", fasta)}, data={"path": job})
        requests.post(f"{self.base}/structure/submit", headers=h, timeout=120,
                      json={"job_name": job, "file_names": ["seqs.fasta"]})

        # poll until every file is done. Transient network hiccups (the hosted
        # status endpoint occasionally read-times-out) are tolerated: sleep and
        # retry rather than crash the whole fold.
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                st = requests.get(f"{self.base}/structure/jobs/{job}/status",
                                  headers=h, timeout=120).json()
            except requests.RequestException:
                time.sleep(self.poll_interval)
                continue
            total = st.get("total_files", 0)
            if total and (st.get("completed", 0) + st.get("failed", 0)) >= total:
                break
            time.sleep(self.poll_interval)

        res = requests.get(
            f"{self.base}/structure/jobs/{job}/files/seqs.fasta/result",
            headers=h, timeout=120).json()
        conf = res.get("confidence_json") or {}
        metrics = res.get("metrics") or {}
        plddt_unit = conf.get("complex_plddt")
        plddt = round(float(plddt_unit) * 100.0, 2) if plddt_unit is not None \
            else round(float(metrics.get("mean_plddt", 0.0)), 2)

        out = {
            "plddt": plddt,
            "iptm": round(float(conf["iptm"]), 4) if tgt and conf.get("iptm") is not None else None,
            "pae": None,  # LiteFold does not expose the PAE matrix
            "n_binder": len(binder),
            "n_target": len(tgt),
        }
        with open(cache, "w") as f:
            json.dump(out, f)
        return out
