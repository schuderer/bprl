import logging
from math import inf
from typing import Dict

from gym_fin.envs import fin_base_sim as f
from gym_fin.envs.sim_env import expose_to_plugins
from gym_fin.envs import utils

logger = logging.getLogger(__name__)


class PensionSimError(Exception):
    pass


class InactiveError(Exception):
    pass


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
        self.reset_called = False
        self.year_fraction = 365 / delta_t

    def run(self):
        if not self.reset_called:
            raise PensionSimError("Simulation must be reset before it can be run")
        while (
            not self.should_stop
            and len([e for e in self.entities if e.active]) > 0
        ):
            self.run_increment()

    def run_increment(self):
        for i, e in enumerate(self.entities):
            if e.active:
                logger.info(f"Entity {i} doing its thing...")
                e.perform_increment()
            else:
                logger.info(f"Skipping inactive Entity {i}...")
        self.time += self.delta_t

    def reset(self):
        """Make this an empty simulation with exactly one
        PensionInsuranceCompany entity and,
        initially, 0 customer entities (`Individual`s).
        """
        # replacing FinBaseSimulation's (example) reset code (no call to super)
        self.reset_called = True
        self.time = 0.0
        self.should_stop = False
        self.entities = [PensionInsuranceCompany(self), PublicOpinion(self)]

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
        my_role: str,
        other: f.Entity,
    ):
        # Some contract specifics could be specified, but we want to
        # experiment with flexible choices here.
        my_number = inf
        my_asset = "eur"
        other_number = inf
        other_asset = "eur"
        super().__init__(me, my_role, my_number, my_asset, other, other_number, other_asset, "cash")


class PensionInsuranceCompany(f.Entity):
    """"Company which provides pension insurance services to `Individual`s."""
    # Not repeating all superclass methods here

    def __init__(self, world: PensionSim):
        super().__init__(world)
        self.resources["cash"] = f.Resource(asset_type="eur", number=20000)
        self.running_cost = 2000

    def check_insurance_contract_request(
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
        # Spend cost to keep doors open
        # can throw away the resulting Resource object)
        r = self.resources["cash"].take(self.running_cost)
        print(f"{self} spending {r}. {self.resources['cash']} left.")
        c: PensionContract
        for c in self.find_contracts(type="insurance"):
            # print(f"insurance contract entities: {c.entities}")
            client = c.entities[1]  # 1 = other, 0 = me
            value: int = self.determine_client_transaction(client)
            if value < 0:
                try:
                    premium = client.request_transfer(-value, "eur", "cash", self)
                    self.resources["cash"].put(premium)
                except f.DeniedError as e:
                    print(f"{self} could not get {-value} from {client}: {e}")
            elif value > 0:
                try:
                    pension = self.resources["cash"].take(value)
                except f.DeniedError as e:
                    print(f"{self} could not take {value} from {self.resources['cash']}: {e}")
                else:
                    try:
                        client.receive_transfer(pension, "cash", self)
                    except f.DeniedError as e:
                        print(f"{self} could not give {pension} to {client}: {e}")
                        self.resources["cash"].put(pension)
                        del(pension)

    def determine_client_transaction(self, client: "Individual") -> int:
        # Static fallback code (replaced by RL actions)
        if client.age < 67:
            # get premium
            return -1500 * self.world.year_fraction
        else:
            # pay out pension
            return 12000 * self.world.year_fraction


class Individual(f.Entity):
    """Natural person with expenses to cover.
    Can receive a salary and/or a pension through a `PensionContract`.
    """

    # Not repeating all superclass methods here
    def __init__(self, world: PensionSim):
        super().__init__(world)
        self.resources["cash"] = f.Resource(asset_type="eur", number=20000)
        self._creation_time = world.time
        self.income = 20000 * self.world.year_fraction
        self.living_expenses = 15000 * self.world.year_fraction

    @property
    def age(self):
        years_passed = int((self.world.time - self._creation_time) / 365)
        return 20 + years_passed

    # @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for an `Individual`"""
        self.ensure_active()

        if self.age >= 67:
            self.income = 0  # Stop working

        if self.world.np_random.uniform() < self.world.year_fraction * utils.cached_cdf(int(self.age / 2) * 2, 85, 10):
            self.active = False  # Die
            return

        # Earn salary
        self.resources["cash"].put(
            f.Resource(asset_type="eur", number=self.income)
        )

        try:
            # Spend living expenses (we won't simulate how this money circulates,
            # can throw away the resulting Resource object)
            r = self.resources["cash"].take(self.living_expenses)
            print(f"{self} spending {r}. {self.resources['cash']} left.")
        except f.DeniedError:
            po = self.world.find_entities(PublicOpinion)[0]
            for c in self.find_contracts(type="insurance"):
                po.accuse(c.entities[1], 100)
            print("Broke")
            self.active = False  # Starve
            return

        if self.resources["cash"].number < self.living_expenses / 2:
            # Be unhappy and tell the world
            po = self.world.find_entities(PublicOpinion)[0]
            for c in self.find_contracts(type="insurance"):
                po.accuse(c.entities[1], 20)

        print(self.age)
        if self.age >= 25 and not self.find_contracts(type="insurance"):
            # Try to get some pension insurance
            print("Trying to get insurance...")
            companies = self.world.find_entities(PensionInsuranceCompany)
            po = self.world.find_entities(PublicOpinion)[0]
            best_companies_first = sorted(companies, key=po.reputation, reverse=True)
            for c in best_companies_first:
                rep = po.reputation(c)
                print(f"    ...Looking at {c} with reputation {rep}...")
                if self.world.np_random.uniform() < self.world.year_fraction * utils.cached_cdf(int(rep / 100) * 100, 0, 1500):
                    try:
                        print("        ...suggesting contract...")
                        contract = PensionContract(me=self, my_role="insured", other=c)
                        # print(f"{self} contracts before: {self.contracts}")
                        c.request_contract(contract)
                        contract.draft = False
                        self.contracts.append(contract)
                        # print(f"{self} contracts after: {self.contracts}")
                        print("            ...got accepted.")
                        break
                    except f.DeniedError:
                        print("            ...got denied.")
                        pass
                else:
                    print("        ...I don't like it.")


class PublicOpinion(f.Entity):
    """"Virtual entity which abstracts a collective of entities.

    In this example, does not strictly need to be derived from `Entity`
    (no funds needed, etc.).
    """

    initial_reputation = 0

    def __init__(self, world: PensionSim):
        super().__init__(world)
        self._reputation = {}

    def accuse(self, offender: f.Entity, severity: int):
        if severity < 0:
            raise ValueError("Severity must be greater than 0")
        if offender.id in self._reputation:
            self._reputation[offender.id] -= severity
        else:
            self._reputation[offender.id] = -severity
        print(f"Reputation of {offender} reduced by {severity} to {self._reputation[offender.id]}")

    def reputation(self, entity: f.Entity):
        return self._reputation.get(entity.id, PublicOpinion.initial_reputation)

    # @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for `PublicOpinion`
        """
        # Slowly recover reputation
        for e_id in self._reputation:
            print(f"Reputation {e_id}: {self._reputation[e_id]} --> {self._reputation[e_id] * 0.9 * self.world.year_fraction}")
            self._reputation[e_id] = int(self._reputation[e_id] * 0.9 * self.world.year_fraction)
            if self._reputation[e_id] > PublicOpinion.initial_reputation - 5 * self.world.year_fraction:
                self._reputation[e_id] = PublicOpinion.initial_reputation
