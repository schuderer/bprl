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
f.ROLES["insured"] = f.Role("insured", "insurer", relation="insurance")
f.ROLES["insurer"] = f.Role("insurer", "insured", relation="insurance")


class PensionContract(f.Contract):
    """Contract between a `PensionInsuranceCompany` and an `Individual`."""

    def __init__(self, me: f.Entity, other: f.Entity):
        super().__init__(my_role="insured", me=me, other=other, reference="cash")


class PensionInsuranceCompany(f.Entity):
    """"Company which provides pension insurance services to `Individual`s."""
    # Not repeating all superclass methods here

    def __init__(self, world: PensionSim):
        super().__init__(world)
        self.resources["cash"] = f.Resource(asset_type="eur", number=20000)

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
            # Spend living expenses (won't need the resulting Resource object)
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


class PublicOpinion(f.Entity):
    """"Virtual entity which abstracts a collective of entities.

    In this example, does not strictly need to be derived from `Entity`
    (no funds needed, etc.).
    """

    def __init__(self, world: PensionSim):
        super().__init__(world)
        self.reputation = {}

    def accuse(self, offender: f.Entity, severity: int):
        if int < 0:
            raise ValueError("Severity must be greater than 0")
        if offender.id in self.reputation:
            self.reputation[offender.id] -= severity
        else:
            self.reputation[offender.id] = severity

    @expose_to_plugins
    def perform_increment(self):
        """Performs one time increment of action(s) for `PublicOpinion`
        """
        # Slowly recover reputation
        for e_id in self.reputation:
            self.reputation[e_id] = int(self.reputation[e_id] * 0.9)
            if self.reputation[e_id] > -5:
                self.reputation[e_id] = 0
