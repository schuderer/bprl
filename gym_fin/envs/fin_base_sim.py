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
# from gym_fin.envs.sim_env import expose_to_plugins
# from gym_fin.envs.sim_env import make_step
from gym_fin.envs.sim_interface import SimulationInterface

logger = logging.getLogger(__name__)


class DeniedError(Exception):
    def __init__(self, message, counter_offer=None):
        super().__init__(message)
        self.counter_offer: "Contract" = counter_offer


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
            cls._roles[role] = obj
            return obj
        else:  # reference to singleton instance
            if role not in cls._roles:
                raise ValueError(f"The role {role} has not been defined.")
            return cls._roles[role]

    @classmethod
    def roles(cls):
        return cls._roles

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(role={self.role}, "
            f"inverse={self.inverse}, relation={self.relation})"
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

        self.id: int = self.create_id()
        self.type: str = my_role_obj.relation
        self.entities: List["Entity"] = [me, other]
        self.roles: List[Role] = [my_role, my_role_obj.inverse]
        self.reference: str = reference
        self.created: float = me.world.time
        self.fulfilled: List[bool] = [False, False]
        self.draft: bool = True
        self.in_dispute: bool = False

    def is_fulfilled(self):
        return all(self.fulfilled)

    def get_inverse(self):
        inverse_contract = copy.deepcopy(self)
        inverse_contract.id = inverse_contract.create_id()
        for attr_name in [
            a for a in dir(inverse_contract) if not a.startswith("_")
        ]:
            attr = getattr(inverse_contract, attr_name)
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
    """A two-party trade contract."""

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
        """Note on perspective:
        E.g. `my_number` is what I am obliged to *give* (not receive),
        and, correspondingly, `other_number` is what the other party has
        to give me.
        """
        super().__init__(me, my_role, other, reference)
        self.numbers: List[float] = [my_number, other_number]
        self.numbers_fulfilled: List[bool] = [False, False]
        self.assets: List[str] = [my_asset, other_asset]

    def to_contract_request(self) -> Dict:
        """Please override this method when inheriting"""
        return {
            "my_role": self.roles[0],
            "other_role": self.roles[1],
            "asked_number": self.numbers[1],
            "asked_asset": self.assets[1],
            "offered_number": self.numbers[0],
            "offered_asset": self.assets[0],
            "reference": self.reference,
        }

    def __repr__(self):
        return (
            f"{'DRAFT ' if self.draft else ''}{self.__class__.__name__}"
            f"(me={self.entities[0]}, my_role={self.roles[0]}, "
            f"other={self.entities[1]}, reference={self.reference}, "
            f"my_number={self.numbers[0]}, my_asset={self.assets[0]}, "
            f"my_number_fulfilled={self.numbers_fulfilled[0]}, "
            f"other_number={self.numbers[1]}, other_asset={self.assets[1]}), "
            f"other_number_fulfilled={self.numbers_fulfilled[1]}"
            f"{' [in dispute]' if self.in_dispute else ''}"
        )


REQUEST_TYPES = [
    "enter_contract",
    "trade",
    "intransfer",
    "outtransfer",
]


ASSET_TYPES = [
    "usd",
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
        self.time: float = 0
        self.delta_t: float = delta_t
        self.entities: List["Entity"] = []
        self.should_stop: bool = False
        self.reset_called: bool = False
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
                logger.debug(f"Entity {i} doing its thing...")
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
        """Signal the simulation to stop at the next chance in the run loop."""
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

        :param asset_type: type of asset (eur, usd, disney shares, etc.)
        :type asset_type: str
        :param number: number of assets (may be fractional)
        :type number: float
        :param allow_negative: allow negative number of assets
        :type allow_negative: bool
        """
        self.id: int = self.create_id()
        self.asset_type: str = asset_type
        self.asset_type_idx: int = ASSET_TYPES.index(asset_type)
        self.number: float = number  # Number of items of asset (amount)
        self.allow_negative: bool = allow_negative

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(asset_type={self.asset_type}, number={self.number}, "
            f"allow_negative={self.allow_negative})"
        )

    def to_obs(self) -> Tuple:
        """Transform the resource object into (part of)
        an AI Gym observation corresponding to `obs_space`.

        The user can use this for their convenience, or
        construct their own observations.
        """
        return (self.asset_type_idx, self.number)

    def obs_space(self):
        """Returns the observation space corresponding to the `to_obs`
        method.
        """
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

        :param number: number of assets to withdraw (may be fractional)
        :type number: float

        :return: A new Resource containing the number of assets withdrawn.

        :raises DeniedError: if negative funds are not allowed but subtracting
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

        :param res: the `Resource` to transfer.
        :type res: Resource

        :raises DeniedError: `asset_type` of `res`'s does not match
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
    """A financial Entity.
    Someone or something (physical or abstract) which can
    take actions in the simulation.
    """

    def __init__(self, world: FinBaseSimulation):
        """Create a financial Entity.

        :param world: The overarching simulation object
        :type world: FinBaseSimulation
        """
        self.id: int = self.create_id()
        self.world: FinBaseSimulation = world
        self.resources: Dict[str, Resource] = {}
        self.contracts: List[Contract] = []
        self.active: bool = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

    def __deepcopy__(self, memo):
        """No copies allowed, only reference"""
        return self

    def ensure_active(self):
        """Raises an EntityInactiveError unless the `Entity` is active.

        :raises EntityInactiveError: `Entity` is inactive
        """
        if not self.active:
            raise EntityInactiveError(
                "Attempting to interact with inactive Entity."
            )

    def find_contracts(
        self, type: str = None, other: "Entity" = None, reference: str = None
    ) -> List[Contract]:
        """Returns filtered list of contracts.

        :param type: relation type of contracts to return
        :type type: optional str, default = any
        :param other: return only contracts with this other `Entity` object
        :type other: optional Entity, default = any
        :param reference: contracts must have this reference
        :type reference: optional str, default = any

        :return: List of matching `Contract` objects
        """
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

    def check_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        """Ask this entity to check (not execute!) a request.
        Usually, the caller of this method is the requesting `Entity`.

        :param request_type: type of the request (used to look up the handling function)
        :type request_type: str
        :param requesting_entity: `Entity` object posing the request (usually a self-reference to the caller)
        :type requesting_entity: Entity
        :param request_contents: information to be used to decide on the request
        :type request_contents: dict

        :raises DeniedError: if request would be denied
        """
        request_func_name = f"check_{request_type}_request"
        request_func = getattr(self, request_func_name)
        request_func(request_type, requesting_entity, request_contents)

    def check_intransfer_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        """Ask an Entity whether we, the `requesting_entity`, may transfer assets to it.
        No DeniedError raised = granted.

        Currently, always grants this kind of request.

        :param requesting_entity: `Entity` object posing the request (usually a self-reference to the caller)
        :type requesting_entity: Entity
        :param request_contents: information to be used to decide on the request, has to include the keys: reference, number, asset_type
        :type request_contents: dict

        :raises DeniedError: if request would be denied
        """
        pass  # always allow for now

    def check_outtransfer_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        """Ask the Entity whether they would transfer assets to the caller (the `requesting_entity`).
        No DeniedError raised = granted.

        :param requesting_entity: `Entity` object posing the request (usually a self-reference to the caller)
        :type requesting_entity: Entity
        :param request_contents: information to be used to decide on the request, has to include the keys: reference, number, asset_type
        :type request_contents: dict

        :raises DeniedError: if request would be denied
        """
        # todo: refactor, trades info needs to be available to agent before check
        # print(f"contracts={self.contracts}")
        possible_contracts = self.find_contracts(
            # type="trade",
            other=requesting_entity,
            reference=request_contents["reference"],
        )
        # print(f"possible_contracts={possible_contracts}")
        payment_contracts = [
            c
            for c in possible_contracts
            if hasattr(c, "numbers") and hasattr(c, "assets")
        ]
        if not payment_contracts:
            raise DeniedError(
                f"No matching contracts for outtransfer by {requesting_entity} with request_contents {request_contents}"
            )
        if len(payment_contracts) > 1:
            raise ValueError(
                f"Found more than one matching contracts for outtransfer "
                f"by {requesting_entity} with request_contents {request_contents}"
            )
        t = payment_contracts[0]
        if t.fulfilled[0]:
            raise DeniedError(f"I ({self}) already fulfilled contract {t}")
        if request_contents["number"] > t.numbers[0]:
            raise DeniedError(
                f"Request to transfer {request_contents['number']} while contract only calls for {t.numbers[0]}"
            )
        if request_contents["asset_type"] != t.assets[0]:
            raise DeniedError(
                f"Request to transfer from asset {request_contents['asset_type']} while contract specifies asset {t.assets[0]}"
            )

    def request_transfer(
        self,
        number: float,
        asset_type: str,
        reference: str,
        requesting_entity: "Entity",
    ) -> Resource:
        """Request the Entity to transfer assets to the caller (the `requesting_entity`).
        No DeniedError raised = transfer succesfully executed.

        :param number: quantity of the asset we ask for
        :type number: float
        :param asset_type: type of the asset (one of `ASSET_TYPES`)
        :type asset_type: str
        :param reference: reference text associated with the resource (needs to match contract)
        :type reference: str
        :param requesting_entity: `Entity` object posing the request (usually a self-reference to the caller)
        :type requesting_entity: Entity

        :return: Resource with this transfer's assets

        :raises DeniedError: transfer has been denied
        """
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
            if hasattr(c, "numbers") and hasattr(c, "assets")
        ]
        # already checked in check_request that there is exactly one contract
        p = payment_contracts[0]
        res = self.resources[reference].take(number)
        p.numbers_fulfilled[0] += number
        p.fulfilled[0] = p.numbers_fulfilled[0] >= p.numbers[0]
        return res

    def receive_transfer(
        self, resource: "Resource", reference: str, requesting_entity: "Entity"
    ):
        """Transfer a resource to the Entity.
        Succesful if no DeniedError is raised.

        :param resource: Resource to transfer. Will be emptied (amount of 0)
        :type resource: Resource
        :param reference: reference text corresponding to resource
        :type reference: str
        :param requesting_entity: `Entity` object posing the request (usually a self-reference to the caller)
        :type requesting_entity: Entity

        :raises DeniedError: transfer has been denied
        """
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
            self.resources[reference].put(resource)
        else:
            # Make a new resource as copy of the resource we were given
            self.resources[reference] = resource.take(resource.number)

    def request_contract(self, contract: Contract):
        """Ask this `Entity` to enter a contract with the `Entity` calling
        `request_contract`.

        :param contract: draft contract
        :type contract: Contract

        :raises DeniedError: contract has been denied
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
        # print(f"{self} contracts before: {self.contracts}")
        self.contracts.append(final_contract)
        # print(f"{self} contracts after: {self.contracts}")

    def dissolve_contract(self, my_or_other_contract: Contract):
        """Ask this `Entity` to disolve a contract

        :param my_or_other_contract: existing contrat
        :type contract: Contract

        :raises DeniedError: contract dissolution has been denied
        """
        # General checks:
        if my_or_other_contract in self.contracts:
            # It's my own version of the contract
            my_contract = my_or_other_contract
            other_entity = my_contract.entities[1]
        else:
            # Other entity's version of our contract
            other_entity = my_or_other_contract.entities[0]
            my_contracts = self.find_contracts(
                type=my_or_other_contract.type,
                other=other_entity,
            )
            if len(my_contracts) == 0:
                raise DeniedError(
                    f"Cannot dissolve non-existing contract {my_or_other_contract}"
                )
            my_contract = my_contracts[0]

        # print(f"##### ME: {self}, MY CONTRACT: {my_contract}")

        self.check_request(
            f"dissolve_{my_contract.type}_contract",
            requesting_entity=other_entity,
            request_contents=my_contract.to_contract_request(),
        )

        # Successful if no DeniedError raised
        # print(f"{self} contracts before: {self.contracts}")
        self.contracts.pop(self.contracts.index(my_contract))
        # print(f"{self} contracts after: {self.contracts}")

    def _transfer_to(
        self,
        number: float,
        asset_type: str,
        reference: str,
        receiving_entity: "Entity",
    ) -> bool:
        """Convenience method to transfer assets
        to another Entity (private method).

        :return: true if successful, false otherwise
        """
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

    def _propose_trade_to(
        self,
        buy_or_sell: str,
        offered_number: float,
        offered_asset: str,
        other: "Entity",
        asked_number: float,
        asked_asset: str,
        reference: str,
    ):
        """Convenience method to propose Trade contract
        to another Entity (private method).

        :return: true if successful, false otherwise
        """
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
            t.draft = False
            self.contracts.append(t)
            return True
        except DeniedError as e:
            logger.info(
                f"{self}'s trade contract was denied by {other}: {e.msg}"
            )
            return False

    # Some test functionality below:
    # @make_step(
    #     observation_space=spaces.Box(
    #         low=np.array(
    #             [-np.inf] * 2 * len(ASSET_TYPES) + [0] * 2 * len(Role.roles())
    #         ),
    #         high=np.array(
    #             [np.inf] * 2 * (len(ASSET_TYPES) + len(Role.roles()))
    #         ),
    #     ),
    #     observation_space_mapping=lambda self: (
    #         self.resources_to_obs()  # + self.relations_to_obs()
    #     ),
    #     action_space=spaces.Discrete(2),
    #     action_space_mapping={0: "A", 1: "B"},
    #     reward_mapping=lambda self: 1 if self.active else 0,
    # )
    def choose_some_action(self):
        """DEMO CODE. Please override/define your own when inheriting.

        Default hard-coded code to choose an action.
        (method runs only if no agent is attached here)
        """
        return "B"

    # @expose_to_plugins
    def perform_increment(self):
        """DEMO CODE. Please override when inheriting.

        Performs one time-increment of actions for this Entity."""
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
