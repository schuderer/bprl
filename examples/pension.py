from math import inf
from typing import Dict

from gym_fin.envs import fin_base_sim as f
from gym_fin.envs.sim_env import expose_to_plugins
from gym_fin import utils


class PensionSim(f.FinBaseSimulation):
    """Main class of an example pension services simulation for demonstrating
    the `fin_env` framework.

    Entity classes:
        - `PensionInsuranceCompany` (only one)
        - `Individual` (0-N)

    The two types of classes can be connected via `PensionContract`s.

    The default time increment is one year (365 days).
    """

    # all inherited methods (even if unchanged) listed here for clarity
    def __init__(self, delta_t: float = 365):
        super().__init__(delta_t)
        # own code here (optional)

    def run(self):
        super().run()
        # own code here (optional)

    def reset(self):
        """Make this an empty simulation with exactly one
        PensionInsuranceCompany entity and,
        initially, 0 customer entities (`Individual`s).
        """
        # replacing FinBaseSimulation's (example) reset code (no call to super)
        self.time = 0.0
        self.should_stop = False
        self.entities = [PensionInsuranceCompany(self)]
        self.public_opinion = PublicOpinion(self)  # virtual entity

    def stop(self):
        super().stop()
        # own code here (optional)


# Extend with domain-specific roles
f.Role("insured", "insurer", relation="insurance")
f.Role("insurer", "insured", relation="insurance")


class PensionContract(f.Trade):
    """Contract between a `PensionInsuranceCompany` and an `Individual`."""

    def __init__(
        self,
        me: f.Entity,
        other: f.Entity,
        reference: str
    ):
        super().__init__(self, my_role="insured", me=me, other=other, reference=reference)
        # Some contract specifics could be specified, but we want to
        # experiment with flexible choices here.
        self.my_number = inf
        self.my_asset = "eur"
        self.other_number = inf
        self.other_asset = "eur"


class PensionInsuranceCompany(f.Entity):
    """"Company which provides pension insurance services to `Individual`s."""
    # Not repeating all superclass methods here

    def __init__(self, world: PensionSim):
        super().__init__(world)
        self.resources["cash"] = f.Resource(asset_type="eur", number=20000)

    def request_insurance_contract(
        self,
        requested_role: str,
        reference: str,
        requesting_entity: f.Entity,
    ):
        request_contents = {
            "reference": reference,
        }
        self.check_request("insurance", requesting_entity, request_contents)
        # Carry out the request. Only technical checks from this point on.
        p = PensionContract(me=self, other=requesting_entity, **request_contents)
        self.contracts.append(p)

    def check_insurance_request(
        self,
        request_type: str,
        requesting_entity: f.Entity,
        request_contents: Dict,
    ):
        pass  # always allow for now

    @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for a
        `PensionInsuranceCompany`
        """
        # todo do stuff
        pass


class Individual(f.Entity):
    """Natural person with expenses to cover.
    Can receive a salary and/or a pension through a `PensionContract`.
    """

    # Not repeating all superclass methods here
    def __init__(self, world: PensionSim):
        super().__init__(world)
        self.resources["cash"] = f.Resource(asset_type="eur", number=20000)
        self.age = 20
        self.income = 20000
        self.living_expenses = 15000

    @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for an `Individual`"""
        if self.age >= 67:
            self.income = 0  # Stop working

        if (self.world.np_random.uniform() < utils.cached_cdf(int(self.age / 2) * 2, 85, 10)):
            self.active = False  # Die
            return

        # Earn salary
        self.resources["cash"].put(
            f.Resource(asset_type="eur", number=self.income)
        )

        try:
            # Spend living expenses (we won't simulate how this money circulates,
            # can throw away the resulting Resource object)
            _ = self.resources["cash"].take(self.living_expenses)
        except f.DeniedError:
            po = self.world.find_entities(PublicOpinion)[0]
            for c in self.find_contracts(type="insurance"):
                po.accuse(c.other, 100)
            self.active = False  # Starve
            return

        if self.resources["cash"].number < self.living_expenses / 2:
            # Be unhappy and tell the world
            po = self.world.find_entities(PublicOpinion)[0]
            for c in self.find_contracts(type="insurance"):
                po.accuse(c.other, 20)

        if self.age >= 25 and not self.find_contracts(type="insurance"):
            # Try to get some pension insurance
            companies = self.world.find_entities(PensionInsuranceCompany)
            po = self.world.find_entities(PublicOpinion)[0]
            best_companies_first = sorted(companies, key=po.reputation, reverse=True)
            for c in best_companies_first:
                rep = po.reputation(c)
                if (self.world.np_random.uniform() < utils.cached_cdf(int(rep / 100) * 100, 0, 1500)):
                    try:
                        c.request_contract(requested_role="insurer", reference="cash", requesting_entity=self)
                        p = PensionContract(me=self, other=c, reference="cash")
                        self.contracts.append(p)
                        break
                    except f.DeniedError:
                        pass


class PublicOpinion(f.Entity):
    """"Virtual entity which abstracts a collective of entities.

    In this example, does not strictly need to be derived from `Entity`
    (no funds needed, etc.).
    """

    def __init__(self, world: PensionSim):
        super().__init__(world)
        self._reputation = {}

    def accuse(self, offender: f.Entity, severity: int):
        if int < 0:
            raise ValueError("Severity must be greater than 0")
        if offender.id in self._reputation:
            self._reputation[offender.id] -= severity
        else:
            self._reputation[offender.id] = -severity

    def reputation(self, entity: f.Entity):
        return self._reputation.get(entity.id, None)

    @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for `PublicOpinion`
        """
        # Slowly recover reputation
        for e_id in self._reputation:
            self._reputation[e_id] = int(self._reputation[e_id] * 0.9)
            if self._reputation[e_id] > -5:
                self._reputation[e_id] = 0
