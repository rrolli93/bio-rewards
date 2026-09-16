"""Flux balance analysis providers for the metabolic verifier.

FBA is a linear program. Maximize c^T v subject to the steady-state mass
balance S v = 0 and the flux bounds lower <= v <= upper. The optimum is
deterministic, so it makes a clean ground truth for scoring a knockout design.

Two providers implement the same three-method protocol:

  * MockFBAProvider   -- a tiny hardcoded toy network solved without any solver,
                         used by the tests and by every offline demo.
  * CobraFBAProvider  -- wraps cobrapy on the bundled E. coli core model, the
                         real use. Optional; raises a clear ImportError if cobra
                         is not installed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

EPS = 1e-6


@runtime_checkable
class FBAProvider(Protocol):
    """The deterministic FBA oracle the reward talks to.

    All fluxes are in the model's own units (for the mock, arbitrary carbon
    units; for cobra, mmol/gDW/h). Knockouts are reaction or gene ids.
    """

    def max_growth(self, knockouts) -> float:
        """Maximum biomass flux achievable with these knockouts (0 if lethal)."""
        ...

    def max_product(self, knockouts, objective, min_growth_frac: float = 0.1) -> float:
        """Maximum product flux while growth stays >= frac of wild-type growth.

        This is the manufacturable-yield question. A design that can only make
        product by refusing to grow is worthless, so growth is constrained first.
        """
        ...

    def coupling(self, knockouts, objective) -> float:
        """Minimum product flux when growth is held at its maximum.

        The growth-coupling metric. If this is > 0 the cell physically cannot
        grow without secreting product, which is the gold standard of strain
        design (selection pressure works for you, not against you).
        """
        ...


# ---------------------------------------------------------------------------
# The toy network behind MockFBAProvider
#
# One carbon precursor pool C and one redox pool N (think NADH). Four reactions:
#
#   GLC_uptake    : (glucose) -> C + N        bound [0, 10]   (the only input)
#   BIOMASS       : C         -> (biomass)    consumes carbon, this is growth
#   EX_target     : N         -> (product)    reoxidizes N, secretes product
#   EX_byproduct  : N         -> (byproduct)  reoxidizes N the "wasteful" way
#
# Steady state gives two balances:
#   carbon:  uptake = biomass                 (all fixed carbon builds biomass)
#   redox:   uptake = product + byproduct     (all N must be reoxidized)
#
# Consequences that make this a real metabolic-engineering toy:
#   * Wild type grows fine and can dump all redox into the byproduct, so at
#     maximum growth the guaranteed product flux (coupling) is ZERO.
#   * Knock out EX_byproduct and the only remaining redox sink is the product,
#     so now product is FORCED whenever the cell grows. Coupling jumps to the
#     full uptake. That is exactly the win a good knockout strategy delivers.
#   * Knock out GLC_uptake or BIOMASS and nothing grows: a lethal design.
# ---------------------------------------------------------------------------


class MockFBAProvider:
    """A dependency-free toy FBA solved by a coarse deterministic flux grid.

    No external solver. The feasible region is enumerated on a fixed grid and
    the linear optima are read off directly, so the answers are exact multiples
    of the grid step and identical on every call.
    """

    R_UPTAKE = "GLC_uptake"
    R_BIOMASS = "BIOMASS"
    R_PRODUCT = "EX_target"
    R_BYPRODUCT = "EX_byproduct"

    UPTAKE_MAX = 10.0
    #: 100 steps across the uptake range gives a 0.1-unit flux grid.
    N_STEPS = 100

    def __init__(self):
        self._step = self.UPTAKE_MAX / self.N_STEPS
        # wild-type growth is a fixed reference, computed once.
        self._wt_growth = self._max_over(self._solve([]), key="g")

    # ---- the tiny LP, enumerated on a grid --------------------------------

    def _bounds(self, knockouts):
        ko = {str(k) for k in knockouts}
        big = self.UPTAKE_MAX
        return {
            "u_max": 0.0 if self.R_UPTAKE in ko else self.UPTAKE_MAX,
            "g_ub": 0.0 if self.R_BIOMASS in ko else big,
            "p_ub": 0.0 if self.R_PRODUCT in ko else big,
            "b_ub": 0.0 if self.R_BYPRODUCT in ko else big,
        }

    def _solve(self, knockouts):
        """Enumerate feasible flux points (u, g, p, b) on the grid.

        Free choices are the uptake u and the product split p; the balances
        pin g = u (carbon) and b = u - p (redox). We keep every grid point
        that respects the bounds. Optima are taken over this set.
        """
        bnd = self._bounds(knockouts)
        pts = []
        n_u = int(round(bnd["u_max"] / self._step))
        for i in range(n_u + 1):
            u = round(i * self._step, 6)
            g = u  # carbon balance: all fixed carbon goes to biomass
            if g > bnd["g_ub"] + EPS:
                continue
            n_p = int(round(u / self._step))
            for j in range(n_p + 1):
                p = round(j * self._step, 6)
                b = round(u - p, 6)  # redox balance
                if p > bnd["p_ub"] + EPS:
                    continue
                if b < -EPS or b > bnd["b_ub"] + EPS:
                    continue
                pts.append({"u": u, "g": g, "p": p, "b": b})
        return pts

    def _flux_of(self, point, objective) -> float:
        """Flux of the reaction named by ``objective`` at a feasible point.

        Only the product and byproduct exchanges carry a settable objective in
        this toy; anything else is structurally not producible (flux 0).
        """
        if objective == self.R_PRODUCT:
            return point["p"]
        if objective == self.R_BYPRODUCT:
            return point["b"]
        return 0.0

    @staticmethod
    def _max_over(points, key) -> float:
        return max((pt[key] for pt in points), default=0.0)

    # ---- the FBAProvider protocol -----------------------------------------

    def max_growth(self, knockouts) -> float:
        return self._max_over(self._solve(knockouts), key="g")

    def max_product(self, knockouts, objective, min_growth_frac: float = 0.1) -> float:
        pts = self._solve(knockouts)
        if not pts:
            return 0.0
        threshold = min_growth_frac * self._wt_growth
        viable = [pt for pt in pts if pt["g"] >= threshold - EPS]
        if not viable:
            return 0.0
        return max(self._flux_of(pt, objective) for pt in viable)

    def coupling(self, knockouts, objective) -> float:
        pts = self._solve(knockouts)
        if not pts:
            return 0.0
        g_max = self._max_over(pts, key="g")
        if g_max <= EPS:
            return 0.0
        at_max = [pt for pt in pts if pt["g"] >= g_max - EPS]
        return min(self._flux_of(pt, objective) for pt in at_max)


#: Named knockout sets that behave differently, for demos and tests.
WILD_TYPE: list[str] = []
KO_BYPRODUCT = [MockFBAProvider.R_BYPRODUCT]  # growth-couples the product
KO_PRODUCT = [MockFBAProvider.R_PRODUCT]      # valid but removes the product sink
KO_LETHAL = [MockFBAProvider.R_UPTAKE]        # starves the cell, no growth


class CobraFBAProvider:
    """Real FBA on the bundled E. coli core model via cobrapy.

    Requires cobra. Install with ``pip install cobra`` or ``uv add cobra``.
    Knockouts may be reaction ids or gene ids; each is resolved against the
    model and knocked out inside a context manager so the base model is never
    mutated.
    """

    def __init__(self, model_name: str = "textbook"):
        try:
            import cobra  # noqa: F401
            from cobra.io import load_model
            from cobra.util.solver import linear_reaction_coefficients
        except ImportError as exc:  # pragma: no cover (exercised only without cobra)
            raise ImportError(
                "CobraFBAProvider needs cobrapy. Install it with "
                "`pip install cobra` or `uv add cobra`, or use MockFBAProvider "
                "for offline runs."
            ) from exc
        self._model = load_model(model_name)
        # Resolve the biomass reaction from the objective coefficients. The
        # objective expression prints as "1.0*Biomass..." so string-splitting it
        # yields the coefficient-prefixed term, not a usable reaction id.
        objective_rxns = list(linear_reaction_coefficients(self._model).keys())
        if not objective_rxns:
            raise ValueError(f"model '{model_name}' has no objective reaction")
        self._biomass = objective_rxns[0].id
        self._wt_growth = self._opt(self._model.slim_optimize())

    @staticmethod
    def _opt(value) -> float:
        """slim_optimize result as a float, mapping infeasible (None or nan) to 0."""
        if value is None or value != value:  # value != value catches nan.
            return 0.0
        return float(value)

    def _knock_out(self, model, knockouts):
        rxn_ids = {r.id for r in model.reactions}
        gene_ids = {g.id for g in model.genes}
        for kid in knockouts:
            kid = str(kid)
            if kid in rxn_ids:
                model.reactions.get_by_id(kid).knock_out()
            elif kid in gene_ids:
                model.genes.get_by_id(kid).knock_out()
            # ids absent from the model are ignored (nothing to disable).

    def max_growth(self, knockouts) -> float:
        with self._model as model:
            self._knock_out(model, knockouts)
            return self._opt(model.slim_optimize())

    def max_product(self, knockouts, objective, min_growth_frac: float = 0.1) -> float:
        with self._model as model:
            self._knock_out(model, knockouts)
            if objective not in {r.id for r in model.reactions}:
                return 0.0
            biomass_rxn = model.reactions.get_by_id(self._biomass)
            biomass_rxn.lower_bound = min_growth_frac * self._wt_growth
            model.objective = objective
            return self._opt(model.slim_optimize())

    def coupling(self, knockouts, objective) -> float:
        with self._model as model:
            self._knock_out(model, knockouts)
            if objective not in {r.id for r in model.reactions}:
                return 0.0
            g_max = self._opt(model.slim_optimize())
            if g_max <= EPS:
                return 0.0
            biomass_rxn = model.reactions.get_by_id(self._biomass)
            # pin growth at its optimum, then find the worst-case product flux.
            biomass_rxn.lower_bound = g_max * (1.0 - 1e-6)
            model.objective = objective
            model.objective_direction = "min"
            return self._opt(model.slim_optimize())
