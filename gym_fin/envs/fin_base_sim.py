# from __future__ import annotations

# Stdlib imports
# from dataclasses import dataclass  # >=3.7
import copy
import logging
from typing import Dict, List, Optional, Tuple, Type

# Third-party imports
from gym import spaces
from gym.utils import seeding
import numpy as np

# Application-level imports
from gym_fin.envs.sim_env import expose_to_plugins
from gym_fin.envs.sim_env import make_step
from gym_fin.envs.sim_interface import SimulationInterface

logger = logging.getLogger(__name__)


class DeniedError(Exception):
    def __init__(self, message, counter_offer=None):
        super().__init__(message)
        self.counter_offer = counter_offer


class EntityInactiveError(Exception):
    pass


class Seq:
    """Base class to create (incremental) class instance ids"""

    seq: int = -1

    @classmethod
    def create_id(cls):
        cls.seq += 1
        return cls.seq


# @dataclass
class Role:
    _roles: Dict[str, "Role"] = {}

    def __new__(
        cls,
        role: str,
        inverse: Optional[str] = None,
        relation: Optional[str] = None,
    ):
        if inverse and relation:  # instantiate new Role
            if role in cls._roles:
                ValueError(f"Cannot redefine already defined role {role}.")
            obj = object.__new__(cls)
            obj.role: str = role
            obj.inverse: str = inverse
            obj.relation: str = relation

            def repr_func(self):
                return (
                    f"{self.__class__.__name__}(my_role={self.my_role}, me={self.me}, "
                    f"other={self.other}, reference={self.reference})"
                )

            obj.__repr__ = repr_func
            cls._roles[role] = obj
            return obj
        else:  # reference to singleton instance
            if role not in cls._roles:
                ValueError(f"The role {role} has not been defined.")
            return cls._roles[role]

    @classmethod
    def roles(cls):
        return cls._roles

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(my_role={self.my_role}, me={self.me}, "
            f"other={self.other}, reference={self.reference})"
        )


# defining roles
Role("buy", "sell", relation="trade")
Role("sell", "buy", relation="trade")


class Contract(Seq):
    """A two-party contract"""

    def __init__(
        self, me: "Entity", my_role: str, other: "Entity", reference: str
    ):
        my_role_obj: Role = Role(my_role)

        self.id = self.create_id()
        self.type = my_role_obj.relation
        self.entities = [me, other]
        self.roles = [my_role, my_role_obj.inverse]
        self.reference = reference
        self.created = me.world.time
        self.fulfilled = [False, False]
        self.draft = True
        self.in_dispute = False

    def is_fulfilled(self):
        return all(self.fulfilled)

    def get_inverse(self):
        inverse_contract = copy.deepcopy(self)
        inverse_contract.create_id()
        for attr_name in [a for a in dir(self) if not a.startswith("_")]:
            attr = getattr(self, attr_name)
            if isinstance(attr, list) and len(attr) == 2:
                attr.reverse()
        return inverse_contract

    def to_contract_request(self) -> Dict:
        """Please override this method when inheriting"""
        return {
            "my_role": self.roles[0],
            "other_role": self.roles[1],
        }

    def __repr__(self):
        return (
            f"{'DRAFT ' if self.draft else ''}{self.__class__.__name__}"
            f"(me={self.entities[0]}, my_role={self.roles[0]}, "
            f"other={self.entities[1]}, reference={self.reference})"
            f"{' [in dispute]' if self.in_dispute else ''}"
        )


class Trade(Contract):
    def __init__(
        self,
        me: "Entity",
        my_role: str,
        my_number: float,
        my_asset: str,
        other: "Entity",
        other_number: float,
        other_asset: str,
        reference: str,
    ):
        super().__init__(me, my_role, other, reference)
        self.numbers = [my_number, other_number]
        self.assets = [my_asset, other_asset]

    def to_contract_request(self) -> Dict:
        """Please override this method when inheriting"""
        return {
            "my_role": self.roles[0],
            "other_role": self.roles[1],
            "asked_number": self.numbers[0],
            "asked_asset": self.assets[1],
            "offered_number": self.numbers[1],
            "offered_asset": self.assets[1],
            "reference": self.reference,
        }

    def __repr__(self):
        return (
            f"{'DRAFT ' if self.draft else ''}{self.__class__.__name__}"
            f"(me={self.entities[0]}, my_role={self.roles[0]}, "
            f"other={self.entities[1]}, reference={self.reference})"
            f"my_number={self.numbers[0]}, my_asset={self.assets[0]}, "
            f"other_number={self.numbers[1]}, other_asset={self.assets[1]})"
            f"{' [in dispute]' if self.in_dispute else ''}"
        )


REQUEST_TYPES = [
    "enter_contract",
    "trade",
    "intransfer",
    "outtransfer",
]


ASSET_TYPES = [
    "eur",
    "s&p500",
]


class FinBaseSimulation(SimulationInterface):
    """Orchestrates the simulation. Does not concern itself with
    observations, rewards, or actions. Those are dealt with by the
    environment-generating functionality (make_step(), get_env(), etc.)
    """

    initial_num_entities = 3

    def __init__(self, delta_t: float = 1.0):
        """Initialize the simulation object.
        This happens separately from resetting the simulation using `reset()`.

        Params:
            - delta_t (float, default 1.0): One day equals 1.0, one
              month equals 30.0, one year equals 365, one hour equals 0.04.
        """
        self.time = 0
        self.delta_t = delta_t
        self.entities = []
        self.should_stop = False
        self.reset_called = False
        self.np_random = None
        self.seed()

    def run(self):
        """The main control flow of the simulation. Only called by the agent
        if it is using callbacks directly.

        NOTE: `reset` has to be called (at least once) before `run` may be called.

        Terminates when an episode is over (does not need to terminate for
        infinite episodes). Use the environment's reset() method to
        abort a possibly running episode and start a new episode.
        Use stop() to just abort an episode without restarting.

        When using get_env(), don't call this method.
        It will then be called automatically by the Env.
        """
        if not self.reset_called:
            raise RuntimeError(
                "`run` called before `reset`. Call `reset` before attempting to `run` the simulation."
            )
        while (
            not self.should_stop
            and len([e for e in self.entities if e.active]) > 0
        ):
            for i, e in enumerate(self.entities):
                logger.info(f"Entity {i} doing its thing...")
                e.perform_increment()
            self.time += self.delta_t

    def reset(self):
        """Return simulation to initial state.
        NOTE: This is NOT the Environment's reset() function.
        Only called be the agent if it is using callbacks directly.

        When using get_env(), don't call this method.
        It will then be called automatically by the Env.
        """
        self.reset_called = True
        self.time = 0.0
        self.should_stop = False
        self.entities = [
            Entity(self) for _ in range(FinBaseSimulation.initial_num_entities)
        ]
        for e in self.entities:
            r = Resource(asset_type="eur", number=100)
            e.resources["cash"] = r

    def stop(self):
        """Signal the simulation to stop at the next chance in the run loop.
        """
        self.should_stop = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __repr__(self):
        return f"{self.__class__.__name__}(delta_t={self.delta_t})"

    def find_entities(
        self, entity_type: Type["Entity"], is_active: bool = True
    ) -> List["Entity"]:
        return [
            e
            for e in self.entities
            if isinstance(e, entity_type) and e.active == is_active
        ]


class Resource(Seq):
    """Resources are things with a certain value. Those can be cash of a
    particular currency, physical things like houses or earrings, or classical
    financial assets like shares or debt obligations. Workforce could also be
    modeled as resource.

    Each `Resource` object may contain more than one "thing", but they must
    all be of the same `asset_type`.

    It is `Resource` objects that are passed around when trading, transferring
    funds or any kind of transaction happens. That way, we make sure that
    no money or other assets are created out of thin air, and whenever we
    Put assets into an account, the original assets are removed. Consequently,
    avoid using plain Python variables for that, but use `Resource` objects.
    """

    def __init__(
        self, asset_type: str, number: float = 0, allow_negative: bool = False
    ):
        """Create a Resource.

        Params:
            - asset_type (str): type of asset (eur, usd, disney shares, etc.)
            - number (float): number of assets (may be fractional)
            - allow_negative (bool): allow negative number of assets
        """
        self.id = self.create_id()
        self.asset_type = asset_type
        self.asset_type_idx = ASSET_TYPES.index(asset_type)
        self.number = number  # Number of items of asset (amount)
        self.allow_negative = allow_negative

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(asset_type={self.asset_type}, number={self.number}, "
            f"allow_negative={self.allow_negative})"
        )

    def to_obs(self) -> Tuple:
        return (self.asset_type_idx, self.number)

    def obs_space(self):
        return spaces.Tuple(
            (
                spaces.Discrete(len(ASSET_TYPES)),
                spaces.Box(
                    low=-np.inf if self.allow_negative else 0,
                    high=np.inf,
                    shape=(),
                ),
            )
        )

    def take(self, number: float) -> "Resource":
        """Take a number of assets out of the `Resource`.

        Params:
            - number (float): number of assets to withdraw (may be fractional)

        Returns:
            A new Resource containing the number of assets withdrawn.

        Throws:
            `DeniedError` if negative funds are not allowed but subtracting
            `number` would lead to negative funds.
        """
        new_number = self.number - number
        if not self.allow_negative and new_number < 0:
            raise DeniedError(
                f"Resource {self.id} must not become negative "
                f"(when subtracting {number} from {self.number})"
            )
        else:
            self.number = new_number
        return Resource(self.asset_type, number)

    def put(self, res: "Resource"):
        """Transfer another `Resource` into this `Resource`. `res` will be
        emptied.

        Params:
            - res (Resource): the `Resource` to transfer.

        Throws:
            DeniedError if `res`'s `asset_type` does not match
            this `Resource`'s `asset_type`.
        """
        if res.asset_type != self.asset_type:
            raise DeniedError(
                f"Asset type {res.asset_type} of Resource "
                f"{res.id} cannot be put into Resource {self.id} "
                f"of asset type {self.asset_type}. Implicit "
                f"Recource conversions are not allowed."
            )
        else:
            # drain original resource and give it chance to raise errors:
            good_to_transfer = res.take(res.number)
            self.number += good_to_transfer.number


class Entity(Seq):
    def __init__(self, world: FinBaseSimulation):
        self.id = self.create_id()
        self.world = world
        self.resources: Dict[str, Resource] = {}
        self.contracts: List[Contract] = []
        self.active = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

    def ensure_active(self):
        if not self.active:
            raise EntityInactiveError(
                "Attempting to interact with inactive Entity."
            )

    def find_contracts(
        self, type: str = None, other: "Entity" = None, reference: str = None
    ) -> List[Contract]:
        return [
            c
            for c in self.contracts
            if (not type or c.type == type)
            and (not other or c.entities[1] is other)
            and (not reference or c.reference == reference)
        ]

    def resources_to_obs(self) -> List:
        li = [0] * len(ASSET_TYPES)
        for key, res in self.resources.items():
            idx = ASSET_TYPES.index(res.asset_type)
            li[idx] += res.number
        return li

    # Todo: change to contracts
    # def relations_to_obs(self):
    #     li = [0] * len(Role.roles())
    #     for key, rel in self.relations.items():
    #         idx = Role.roles().keys().index(rel.role.name)
    #         li[idx] += 1  # TODO: could add up relations' wealth as well
    #     return li

    # @make_step(
    #     observation_space=spaces.Tuple(
    #         (
    #             spaces.Box(
    #                 low=np.array(
    #                     [-np.inf] * 2 * len(ASSET_TYPES) + [0] * 2 * len(ROLES)
    #                 ),
    #                 high=np.array(
    #                     [np.inf] * 2 * (len(ASSET_TYPES) + len(ROLES))
    #                 ),
    #             ),
    #             spaces.Discrete(len(REQUEST_TYPES)),
    #             spaces.Discrete(len(ROLES)),
    #         )
    #     ),
    #     observation_space_mapping=lambda self, request_type, requesting_entity, request_contents: (
    #         (
    #             self.resources_to_obs()
    #             + requesting_entity.resources_to_obs()
    #             + self.relations_to_obs()
    #             + requesting_entity.relations_to_obs()
    #         ),
    #         REQUEST_TYPES.index(request_type),
    #         ROLES.keys().index(request_contents["role_name"])
    #         if "role_name" in request_contents
    #         else 0,
    #     ),
    #     action_space=spaces.Discrete(2),
    #     action_space_mapping={0: None, 1: DeniedError},
    #     reward_mapping=lambda self, request_type, requesting_entity, request_contents: sum(
    #         0  # self.resources_to_obs()
    #     ),
    # )
    def check_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        request_func_name = f"check_{request_type}_request"
        request_func = getattr(self, request_func_name)
        request_func(request_type, requesting_entity, request_contents)

    def check_intransfer_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        pass  # always allow for now

    def check_outtransfer_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        # todo: refactor, trades info needs to be available to agent before check
        possible_contracts = self.find_contracts(
            # type="trade",
            other=requesting_entity,
            reference=request_contents["reference"],
        )
        payment_contracts = [
            c
            for c in possible_contracts
            if hasattr(c, "numbers") and hasattr(c, "assets")
        ]
        if not payment_contracts:
            DeniedError(
                f"No matching contracts for outtransfer by {requesting_entity} with request_contents {request_contents}"
            )
        if len(payment_contracts) > 2:
            ValueError(
                f"Found more than one matching contracts for outtransfer "
                f"by {requesting_entity} with request_contents {request_contents}"
            )
        t = payment_contracts[0]
        if t.fulfilled[0]:
            raise DeniedError(f"I ({self}) already fulfilled contract {t}")
        if request_contents.number > t.numbers[0]:
            raise DeniedError(
                f"Request to transfer {request_contents.number} while contract only calls for {t.numbers[0]}"
            )
        if request_contents.asset_type != t.assets[0]:
            raise DeniedError(
                f"Request to transfer from asset {request_contents.asset_type} while contract specifies asset {t.assets[0]}"
            )

    def request_transfer(
        self,
        number: float,
        asset_type: str,
        reference: str,
        requesting_entity: "Entity",
    ) -> Resource:
        self.check_request(
            "outtransfer",
            requesting_entity,
            {
                "number": number,
                "asset_type": asset_type,
                "reference": reference,
            },
        )
        # Carry out the request. Only technical checks from this point on.
        if reference not in self.resources:
            # TODO: smarter way to reference resources?
            raise ValueError(f"No resource with key '{reference}'")
        # TODO: refactor code duplication
        possible_contracts = self.find_contracts(
            other=requesting_entity, reference=reference
        )
        payment_contracts = [
            c
            for c in possible_contracts
            if hasattr(c, "my_number") and hasattr(c, "my_asset")
        ]
        # already checked in check_request that there is exactly one contract
        p = payment_contracts[0]
        res = self.resources[reference].take(number)
        p.numbers[0] = p.numbers[0] - number
        p.fulfilled[0] = p.numbers[0] == 0
        return res

    def receive_transfer(
        self, resource: "Resource", reference: str, requesting_entity: "Entity"
    ):
        self.check_request(
            "intransfer",
            requesting_entity,
            {
                "number": resource.number,
                "asset_type": resource.asset_type,
                "reference": reference,
            },
        )
        # Carry out the request. Only technical checks from this point on.
        if reference in self.resources:
            # TODO: smarter way to reference resources?
            self.resource[reference].put(resource)
        else:
            # Make a new resource as copy of the resource we were given
            self.resource[reference] = resource.take(resource.number)

    def request_contract(self, contract: Contract):
        """Ask this `Entity` to enter a contract with the `Entity` calling
        `request_contract`.

        Params:
            - contract: draft contract
        """
        # General checks:
        if not contract.draft:
            raise DeniedError(
                f"Cannot request a definite (non-draft) contract {contract}"
            )
        if contract.entities[1] is not self:
            raise DeniedError(f"{self} must be a party in contract {contract}")
        type = contract.type
        # print(f"{check_func_name}({self}, {requested_role}, {reference}, {requesting_entity})")

        self.check_request(
            f"{type}_contract",
            requesting_entity=contract.entities[0],
            request_contents=contract.to_contract_request(),
        )

        # Successful if no DeniedError raised
        final_contract = contract.get_inverse()
        final_contract.draft = False
        self.contracts.append(final_contract)

    # On from handling requests to actually initiating stuff:
    def transfer_to(
        self,
        number: float,
        asset_type: str,
        reference: str,
        receiving_entity: "Entity",
    ) -> bool:
        res = self.resources[reference].take(number)
        try:
            receiving_entity.receive_transfer(res, reference, self)
            return True
        except DeniedError as e:
            logger.info(
                f"{self}'s transfer was denied by "
                f"{receiving_entity}: {e.msg}"
            )
            self.resources[reference].put(res)
            return False

    def propose_trade_to(
        self,
        buy_or_sell: str,
        offered_number: float,
        offered_asset: str,
        other: "Entity",
        asked_number: float,
        asked_asset: str,
        reference: str,
    ):
        t = Trade(
            me=self,
            my_role=buy_or_sell,
            other=other,
            my_number=offered_number,
            my_asset=offered_asset,
            other_number=asked_number,
            other_asset=asked_asset,
        )
        try:
            other.request_contract(t)
            self.contracts.append(t)
            return True
        except DeniedError as e:
            logger.info(
                f"{self}'s trade contract was denied by {other}: {e.msg}"
            )
            return False

    # Some test functionality below:
    @make_step(
        observation_space=spaces.Box(
            low=np.array(
                [-np.inf] * 2 * len(ASSET_TYPES) + [0] * 2 * len(Role.roles())
            ),
            high=np.array(
                [np.inf] * 2 * (len(ASSET_TYPES) + len(Role.roles()))
            ),
        ),
        observation_space_mapping=lambda self: (
            self.resources_to_obs()  # + self.relations_to_obs()
        ),
        action_space=spaces.Discrete(2),
        action_space_mapping={0: "A", 1: "B"},
        reward_mapping=lambda self: 1 if self.active else 0,
    )
    def choose_some_action(self):
        # custom if-then-else code to act on
        # (method runs only if no agent is attached here)
        return "B"

    @expose_to_plugins
    def perform_increment(self):
        """Performs one time-increment of actions for this Entity"""
        decision = self.choose_some_action()
        if self.active:
            if decision == "A":
                logger.info("doing A, doing fine.")
            elif decision == "B":
                logger.info("doing B. I've been eaten by a grue.")
                self.active = False
            else:
                raise ValueError("Illegal decision, only 'A' or 'B' allowed.")
        else:
            logger.info("...still being digested by the grue...")
