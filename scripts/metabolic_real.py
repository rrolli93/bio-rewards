"""Real FBA on the E. coli core textbook model via CobraFBAProvider.

Run with:

    uv run --with cobra python scripts/metabolic_real.py

The metabolic-engineering example: maximize acetate secretion (EX_ac_e) and knock
out cytochrome oxidase (CYTBD), the aerobic-respiration route that reoxidizes NADH
without secreting anything. With respiration intact the cell can hit maximum growth
while dumping zero acetate, so acetate is not growth-coupled. Remove that competing
redox sink and fermentative acetate becomes the only way to balance redox at high
growth, so acetate turns growth-coupled. That shift is exactly what the reward
rewards, and it moves the score.
"""

from biorewards.verifiers.metabolic import MetabolicReward
from biorewards.verifiers.metabolic.providers import CobraFBAProvider

PRODUCT = "EX_ac_e"          # acetate exchange.
COMPETING_KO = ["CYTBD"]     # cytochrome oxidase, the aerobic redox sink.


def main():
    provider = CobraFBAProvider("textbook")

    # Anchors sized to the textbook acetate scale (max acetate ~18.7 mmol/gDW/h,
    # coupled acetate under the knockout ~8.5). Set from a reference panel in a
    # real deployment; here they are fixed so the demo is reproducible.
    reward = MetabolicReward(
        provider=provider,
        product_good=20.0,
        coupling_good=10.0,
    )

    wt_growth = provider.max_growth([])
    print("E. coli core (textbook), objective = acetate secretion EX_ac_e")
    print(f"  wild-type growth                    {wt_growth:.4f} /h")
    print()

    wt = {"knockouts": [], "objective": PRODUCT}
    ko = {"knockouts": COMPETING_KO, "objective": PRODUCT}

    for label, design in (("wild type", wt), ("knock out CYTBD", ko)):
        max_prod = provider.max_product(design["knockouts"], PRODUCT)
        coupled = provider.coupling(design["knockouts"], PRODUCT)
        growth = provider.max_growth(design["knockouts"])
        result = reward.evaluate(design)
        print(f"{label} (knockouts={design['knockouts']}):")
        print(f"  max growth                          {growth:.4f} /h")
        print(f"  max acetate flux (>=10% growth)     {max_prod:.4f} mmol/gDW/h")
        print(f"  guaranteed acetate at max growth    {coupled:.4f} mmol/gDW/h  (coupling)")
        print(f"  reward                              {result.reward}")
        print()

    print("Knocking out the competing redox sink (CYTBD) growth-couples acetate")
    print("(coupling 0 -> ~8.5), which raises the growth_coupling component and")
    print("changes the reward. FBA is the deterministic ground truth throughout.")


if __name__ == "__main__":
    main()
