"""Live fold + score of the C1-01 scaffold against KISS1R via the binder verifier.

Binder = YNWNSFGLRF, the canonical backbone of C1-01 (the Eurofins Round 1
KISS1R candidate {D-Tyr}NWNSFGL{Arg(Me)}F-NH2). The D-Tyr and Nomega-methyl-Arg
modifications are NOT represented in this fold (Boltz takes canonical residues);
this scores the KP-10-like scaffold and pose against the real receptor.

Target = human KISS1R (UniProt Q969F8).

Real GPU job via boltz-api. Needs BOLTZ_API_KEY in the environment.
"""

import json

from biorewards.verifiers.binder import BinderReward
from biorewards.verifiers.binder.providers import BoltzApiFoldProvider

KISS1R = (
    "MHTVATSGPNASWGAPANASGCPGCGANASDGPVPSPRAVDAWLVPLFFAALMLLGLVGN"
    "SLVIYVICRHKPMRTVTNFYIANLAATDVTFLLCCVPFTALLYPLPGWVLGDFMCKFVNY"
    "IQQVSVQATCATLTAMSVDRWYVTVFPLRALHRRTPRLALAVSLSIWVGSAAVSAPVLAL"
    "HRLSPGPRAYCSEAFPSRALERAFALYNLLALYLLPLLATCACYAAMLRHLGRVAVRPAP"
    "ADSALQGQVLAERAGAVRAKVSRLVAAVVLLFAACWGPIQLFLVLQALGPAGSWHPRSYA"
    "AYALKTWAHCMSYSNSALNPLLYAFLGSHFRQAFRRVCPCAPRRPRRPRRPGPSDPAAPH"
    "AELLRLGSHPAPARAQKPGSSGLAARGLCVLGEDNAPL"
)

x = {"sequence": "YNWNSFGLRF", "target": KISS1R}

print("Folding C1-01 scaffold (YNWNSFGLRF) against KISS1R via boltz-api ...")
rf = BinderReward(fold_provider=BoltzApiFoldProvider(num_samples=1))
res = rf.evaluate(x)

print("\n=== RESULT ===")
print("status:", res.status, " reward:", res.reward)
for c in res.components:
    print(f"  {c.name:16s} raw={str(c.raw):>8}  score={c.score}  ({c.detail})")
print("\napplicability:", res.applicability.reason)

with open("kiss1r_c101_result.json", "w") as f:
    json.dump(res.as_dict(), f, indent=2)
print("\nwrote kiss1r_c101_result.json")
