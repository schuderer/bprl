# from __future__ import annotations

# Stdlib imports
# from dataclasses import dataclass  # >=3.7
import logging
from typing import Dict, List

# Third-party imports
from gym import spaces
import numpy as np

# Application-level imports
from gym_fin.envs.sim_env import expose_to_plugins
from gym_fin.envs.sim_env import make_step
from gym_fin.envs.sim_interface import SimulationInterface

logger = logging.getLogger(__name__)


class DeniedError(Exception):
    pass


# @dataclass
class Role:
    def __init__(self, role: str, inverse: str, relation: str):
        self.role: str = role
        self.inverse: str = inverse
        self.relationship: str = relation


class Contract:
    """A two-party contract"""

    def __init__(
        self, my_role: str, me: "Entity", other: "Entity", reference: str
    ):
        self.role: str = ROLES[my_role]
        self.type = self.role.relation
        self.me: str = me
        self.other: str = other
        self.role_me: str = my_role
        self.my_role: str = self.role_me  # for usability's sake
        self.role_other: str = self.roles.inverse
        self.reference = reference
        self.created = me.world.time
        self.fulfilled_by_me = False
        self.fulfilled_by_other = False
        self.in_dispute = False

    @property
    def fulfilled(self):
        return self.fulfilled_by_me and self.fulfilled_by_other


ROLES = {
    "buy": Role("buy", "sell", relation="trade"),
    "sell": Role("sell", "buy", relation="trade"),
}


class Trade(Contract):
    def __init__(
        self,
        me: "Entity",
        other: "Entity",
        my_number: float,
        my_asset: str,
        other_number: float,
        other_asset: str,
    ):
        super().__init__(self, my_role="buy", me=me, other=other)
        self.my_number = my_number
        self.my_asset = my_asset
        self.other_number = other_number
        self.other_asset = other_asset
        # self.status = 'open'  # open, denied, accepted, retracted, fulfilled


REQUEST_TYPES = [
    "enter_contract",
    "trade",
    "in_transfer",
    "out_transfer",
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
        """Initialize the simulation.

        Params:
            - delta_t (float, default 1.0): One day equals 1.0, one
              month equals 30.0, one year equals 360, one hour equals 0.04.
        """
        self.entities = []
        self.should_stop = False
        self.delta_t = delta_t

    def run(self):
        """The main control flow of the simulation. Only called by the agent
        if it is using callbacks directly.

        Terminates when an episode is over (does not need to terminate for
        infinite episodes). Use the environment's reset() method to
        abort a possibly running episode and start a new episode.
        Use stop() to just abort an episode without restarting.

        When using get_env(), don't call this method.
        It will then be called automatically by the Env.
        """
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
        Only this function if your agent is using callbacks directly.

        When using get_env(), don't call this method.
        It will then be called automatically by the Env.
        """
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


class Seq:
    """Base class to create (incremental) class instance ids"""

    seq: int = -1

    @classmethod
    def create_id(cls):
        cls.seq += 1
        return cls.seq


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
            f"Resource(asset_type={self.asset_type}, number={self.number}, "
            f"allow_negative={self.allow_negative})"
        )

    def to_obs(self):
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

    def take(self, number: float):
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
        self.resources: Dict[Resource] = {}
        self.contracts: List[Contract] = []
        self.active = True

    def __repr__(self):
        return f"Entity({self.id})"

    def find_contracts(
        self, type: str = None, other: "Entity" = None, reference: str = None
    ):
        return [
            c
            for c in self.contracts
            if (not type or c.type == type)
            and (not other or c.other is other)
            and (not reference or c.reference == reference)
        ]

    def resources_to_obs(self):
        li = [0] * len(ASSET_TYPES)
        for key, res in self.resources.items():
            idx = ASSET_TYPES.index(res.asset_type)
            li[idx] += res.number
        return li

    def relations_to_obs(self):
        li = [0] * len(ROLES)
        for key, rel in self.relations.items():
            idx = ROLES.keys().index(rel.role.name)
            li[idx] += 1  # TODO: could add up relations' wealth as well
        return li

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
        if request_type == "trade":
            # always allow for now
            pass
        elif request_type == "in_transfer":
            # always allow
            pass
        elif request_type == "out_transfer":
            # todo: refactor, trades info needs to be available to agent before check
            matching_trades = self.find_contracts(
                type="trade",
                other=requesting_entity,
                reference=request_contents.reference,
            )
            if not matching_trades:
                DeniedError(
                    f"No matching trades for out_transfer by {requesting_entity} with request_contents {request_contents}"
                )
            if len(matching_trades) > 2:
                ValueError(
                    f"Found more than one matching trades for out_transfer by {requesting_entity} with request_contents {request_contents}"
                )
            t = matching_trades[0]
            if t.fulfilled_by_me:
                raise DeniedError(f"{self} already fulfilled Trade {t}")
            if request_contents.number > t.my_number:
                raise DeniedError(
                    f"Requesting to transfer {request_contents.number} while contract only calls for {t.my_number}"
                )
        else:
            raise ValueError(f"Unknown request_type: {request_type}")

    def request_transfer(
        self,
        number: float,
        asset_type: str,
        reference: str,
        requesting_entity: "Entity",
    ):
        self.check_request(
            "out_transfer",
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
            raise ValueError(f"No resource with key {reference}")
        # TODO: refactor code duplication
        matching_trades = self.find_contracts(
            type="trade", other=requesting_entity, reference=reference
        )
        t = matching_trades[0]  # already checked that there is one
        res = self.resources[reference].take(number)
        t.my_number = t.my_number - number
        t.fulfilled_by_me = t.my_number == 0
        return res

    def receive_transfer(
        self, resource: "Resource", reference: str, requesting_entity: "Entity"
    ):
        self.check_request(
            "in_transfer",
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

    def request_trade(
        self,
        offered_number: float,
        offered_asset: str,
        asked_number: float,
        asked_asset: str,
        reference: str,
        requesting_entity: "Entity",
    ):
        request_contents = {
            "my_number": asked_number,
            "my_asset": asked_asset,
            "other_number": offered_number,
            "other_asset": offered_asset,
            "reference": reference,
        }
        self.check_request("trade", requesting_entity, request_contents)
        # Carry out the request. Only technical checks from this point on.
        t = Trade(me=self, other=requesting_entity, **request_contents)
        self.contracts.append(t)

    # On from handling requests to actually initiating stuff:
    def transfer_to(
        self,
        number: float,
        asset_type: str,
        reference: str,
        receiving_entity: "Entity",
    ):
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
        offered_number: float,
        offered_asset: str,
        asked_number: float,
        asked_asset: str,
        reference: str,
        other: "Entity",
    ):
        request_contents = {
            "my_number": offered_number,
            "my_asset": offered_asset,
            "other_number": asked_number,
            "other_asset": asked_asset,
            "reference": reference,
        }
        t = Trade(me=self, other=other, **request_contents)
        try:
            other.request_trade(
                offered_number=offered_number,  # todo: cumbersome
                offered_asset=offered_asset,
                asked_number=asked_number,
                asked_asset=asked_asset,
                reference=reference,
                requesting_entity=self,
            )
            self.contracts.append(t)
            return True
        except DeniedError as e:
            logger.info(
                f"{self}'s trade contract was " f"denied by {other}: {e.msg}"
            )
            return False

    # Some test functionality below:
    @make_step(
        observation_space=spaces.Box(
            low=np.array(
                [-np.inf] * 2 * len(ASSET_TYPES) + [0] * 2 * len(ROLES)
            ),
            high=np.array([np.inf] * 2 * (len(ASSET_TYPES) + len(ROLES))),
        ),
        observation_space_mapping=lambda self: (
            self.resources_to_obs() + self.relations_to_obs()
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
