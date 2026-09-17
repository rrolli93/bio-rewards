"""A/B the same binder+target complex across two real folders.

boltz-api (cached from the earlier run) exposes PAE, so it scores the full set
ipTM + ipSAE + pae_interaction + pLDDT. LiteFold does NOT expose PAE, so it scores
ipTM + pLDDT only. Same verifier, same candidate, two instruments.
"""

from biorewards.verifiers.binder import BinderReward
from biorewards.verifiers.binder.providers import BoltzApiFoldProvider, LiteFoldFoldProvider

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


def show(label, provider):
    res = BinderReward(fold_provider=provider).evaluate(x)
    fold = {c.name: (c.raw, str(c.score)) for c in res.components}
    print(f"\n=== {label} ===  reward {res.reward}  ({res.status})")
    for name in ("plddt", "iptm", "ipsae", "pae_interaction"):
        if name in fold:
            print(f"  {name:16s} raw={str(fold[name][0]):>8}  score={fold[name][1][:6]}")
        else:
            print(f"  {name:16s} (not provided by this folder)")


print("Folding C1-01 scaffold vs KISS1R on two folders ...")
show("boltz-api (cached, has PAE)", BoltzApiFoldProvider(num_samples=1))
show("LiteFold (live, no PAE)", LiteFoldFoldProvider())
print("\nSame verifier, same candidate. boltz-api scores the full interface set;")
print("LiteFold scores ipTM + pLDDT because it does not return the PAE matrix.")
