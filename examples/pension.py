import logging
from math import inf
from typing import Dict

from gym_fin.envs import fin_base_sim as f
from gym_fin.envs.sim_env import expose_to_plugins
from gym_fin.envs import utils

logger = logging.getLogger(__name__)


class PensionSimError(Exception):
    pass


class PensionSim(f.FinBaseSimulation):
    """Main class of an example pension services simulation for demonstrating
    the `fin_env` framework.

    Entity classes:
        - `PensionInsuranceCompany` (only one)
        - `Individual` (0-N)
        - `PublicOpinion` (exactly one)

    The two types of classes can be connected via `PensionContract`s.

    The default time increment is one year (365 days).
    """

    # all inherited methods (even if unchanged) listed here for clarity
    def __init__(self, delta_t: float = 365, max_individuals: int = 100, max_days: int = -1):
        super().__init__(delta_t)
        # own code here (optional)
        self.reset_called = False
        self.year_fraction = delta_t / 365
        self.max_individuals = max_individuals
        self.max_days = max_days  # max_steps * delta_t

    def run(self, max_t: int = -1):
        if not self.reset_called:
            raise PensionSimError("Simulation must be reset before it can be run")
        while (
            not self.should_stop
            # and len([e for e in self.entities if e.active]) > 1
            and len([e for e in self.entities if isinstance(e, PensionInsuranceCompany) and e.active]) >= 1
            and (max_t == -1 or max_t > self.time)
        ):
            self.run_increment()

    def run_increment(self):
        for i, e in enumerate(self.entities):
            if e.active:
                logger.info(f"Entity {i} doing its thing at time {self.time}...")
                e.perform_increment()
            else:
                logger.info(f"Skipping inactive Entity {i} at time {self.time}...")
        self.time += self.delta_t
        self.max_out_individuals()
        # print(self.time)
        if self.max_days > 0 and self.time >= self.max_days:
            # print(f"TIME UP: {self.time} > {self.max_days}")
            self.stop()

    def reset(self):
        """Make this an empty simulation with exactly one
        PensionInsuranceCompany entity, one PublicOpinion entity and,
        initially, 0 customer entities (`Individual`s).
        """
        # replacing FinBaseSimulation's (example) reset code (no call to super)
        self.reset_called = True
        self.time = 0.0
        self.should_stop = False
        self.entities = [PensionInsuranceCompany(self), PublicOpinion(self)]
        self.max_out_individuals()

    def stop(self):
        super().stop()
        # own code here (optional)

    def max_out_individuals(self):
        ind = self.find_entities(Individual, is_active=True)
        new_ind = [Individual(self) for _ in range(self.max_individuals - len(ind))]
        self.entities.extend(new_ind)


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

    def check_dissolve_insurance_contract_request(
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
        logger.debug(f"{self} performing increment...")

        # Spend cost to keep doors open
        # can throw away the resulting Resource object)
        try:
            r = self.resources["cash"].take(self.running_cost)
        except f.DeniedError as e:
            logger.info(f"{self} could not spend running cost {self.running_cost}: {e}")
            self.active = False
            return

        logger.info(f"{self} spent {r}. {self.resources['cash']} left.")
        c: PensionContract
        for c in self.find_contracts(type="insurance"):
            # print(f"insurance contract entities: {c.entities}")
            client = c.entities[1]  # 1 = other, 0 = me
            if client.active:
                value: int = self.determine_client_transaction(client)
                if value < 0:
                    try:
                        premium = client.request_transfer(-value, "eur", "cash", self)
                        self.resources["cash"].put(premium)
                    except f.DeniedError as e:
                        logger.info(f"{self} could not get {-value} from {client}: {e}")
                elif value > 0:
                    try:
                        pension = self.resources["cash"].take(value)
                    except f.DeniedError as e:
                        logger.info(f"{self} could not take {value} from {self.resources['cash']}: {e}")
                        self.active = False
                        return
                    else:
                        try:
                            client.receive_transfer(pension, "cash", self)
                        except f.DeniedError as e:
                            logger.info(f"{self} could not give {pension} to {client}: {e}")
                            self.resources["cash"].put(pension)
                            del(pension)

    def determine_client_transaction(self, client: "Individual") -> int:
        print("###############################################")
        print("###    STATIC DECISION FUNCTION CALLED     ####")
        print("###   (SHOULD NOT APPEAR WHEN LEARNING)    ####")
        print("###############################################")
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
        years_passed = int((self.world.time - self._creation_time) / self.world.delta_t)
        return 20 + years_passed

    def check_dissolve_insurance_contract_request(
        self,
        request_type: str,
        requesting_entity: f.Entity,
        request_contents: Dict,
    ):
        pass  # always allow for now

    def die(self):
        # Dissolve contract
        for c in self.find_contracts(type="insurance"):
            c.entities[1].dissolve_contract(c)
            self.dissolve_contract(c)
        # Starve
        self.active = False

    # @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for an `Individual`"""
        logger.debug(f"{self} performing increment...")
        self.ensure_active()

        if self.age >= 67:
            self.income = 0  # Stop working
            logger.debug(f"{self} retired at age {self.age}")

        if self.world.np_random.uniform() < self.world.year_fraction * utils.cached_cdf(int(self.age / 2) * 2, 85, 10):
            self.die()
            logger.debug(f"{self} died at age {self.age}")
            return

        # Earn salary
        self.resources["cash"].put(
            f.Resource(asset_type="eur", number=self.income)
        )

        try:
            # Spend living expenses (we won't simulate how this money circulates,
            # can throw away the resulting Resource object)
            r = self.resources["cash"].take(self.living_expenses)
            logger.info(f"{self} spending {r}. {self.resources['cash']} left.")
        except f.DeniedError:
            po = self.world.find_entities(PublicOpinion)[0]
            for c in self.find_contracts(type="insurance"):
                po.accuse(c.entities[1], 100)
            logger.info(f"{self} starved at age {self.age}")
            self.die()
            return

        if self.resources["cash"].number < self.living_expenses / 2:
            # Be unhappy and tell the world
            po = self.world.find_entities(PublicOpinion)[0]
            for c in self.find_contracts(type="insurance"):
                po.accuse(c.entities[1], 20)

        logger.info(self.age)
        # DEBUG:
        old_contracts = self.find_contracts(type="insurance")
        if self.age >= 25 and not old_contracts:
            # Try to get some pension insurance
            logger.info("Trying to get insurance...")
            companies = self.world.find_entities(PensionInsuranceCompany)
            po = self.world.find_entities(PublicOpinion)[0]
            best_companies_first = sorted(companies, key=po.reputation, reverse=True)
            for c in best_companies_first:
                rep = po.reputation(c)
                logger.info(f"    ...Looking at {c} with reputation {rep}...")
                if self.world.np_random.uniform() < utils.cached_cdf(int(rep / 100) * 100, 0, 100):
                    try:
                        logger.info("        ...suggesting contract...")
                        contract = PensionContract(me=self, my_role="insured", other=c)
                        # print(f"{self} contracts before: {self.contracts}")
                        c.request_contract(contract)
                        contract.draft = False
                        self.contracts.append(contract)
                        # DEBUG -- REMOVE
                        if len(self.contracts) > 1:
                            raise RuntimeError(
                                f"self.contracts of {self} is greater than 1: {self.contracts}. "
                                f"Somehow had an old contract already: {self.contracts[0]} while I thought it had {old_contracts}"
                            )
                        # print(f"{self} contracts after: {self.contracts}")
                        logger.info("            ...got accepted.")
                        break
                    except f.DeniedError:
                        logger.info("            ...got denied.")
                        pass
                else:
                    logger.info("        ...I don't like it.")


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
        logger.info(f"Reputation of {offender} reduced by {severity} to {self._reputation[offender.id]}")

    def reputation(self, entity: f.Entity):
        return self._reputation.get(entity.id, PublicOpinion.initial_reputation)

    # @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for `PublicOpinion`
        """
        logger.debug(f"{self} performing increment...")
        # Slowly recover reputation
        for e_id in self._reputation:
            logger.debug(f"Reputation {e_id}: {self._reputation[e_id]} --> {self._reputation[e_id] * (0.95 / self.world.year_fraction)}")
            self._reputation[e_id] = int(self._reputation[e_id] * (0.95 / self.world.year_fraction))
            if self._reputation[e_id] > PublicOpinion.initial_reputation - 5 * self.world.year_fraction:
                self._reputation[e_id] = PublicOpinion.initial_reputation
