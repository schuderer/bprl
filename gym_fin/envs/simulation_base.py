# from __future__ import annotations
# from dataclasses import dataclass
import logging
from typing import Dict, Optional

from gym import spaces
import numpy as np

from gym_fin.envs.sim_env import make_step
from gym_fin.envs.simulation_interface import SimulationInterface

logger = logging.getLogger(__name__)


class DeniedError(Exception):
    pass


# @dataclass
class Role:
    def __init__(self, name: str, inverse: Optional[str] = None):
        self.name: str = name
        # inverse = (buyer/seller, etc., None for none
        # (symmetries like friend/friend would be reciprocal but it's own inverse))
        self.inverse: Optional[str] = inverse


# @dataclass
class Relation:
    def __init__(self, role: Role, entity: "Entity"):
        self.role: str = role
        # inverse = (buyer/seller, etc., None for none
        # (symmetries like friend/friend would be reciprocal but it's own inverse))
        self.entity: Optional[str] = entity


ROLES = {
    "client": Role("client", "supplier"),
    "supplier": Role("supplier", "client"),
    "owner": Role("owner", "property"),
    "property": Role("property", "owner"),
}


REQUEST_TYPES = [
    "relation",
    "transfer",
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

    initial_num_entities = 5

    def __init__(self):
        self.entities = []
        self.shouldStop = False

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
            not self.shouldStop
            and len([e for e in self.entities if e.active]) > 0
        ):
            for i, e in enumerate(self.entities):
                logger.info("Entity {} doing its thing...".format(i))
                e.perform_increment()

    def reset(self):
        """Return simulation to initial state.
        NOTE: This is NOT the Environment's reset() function.
        Only this function if your agent is using callbacks directly.

        When using get_env(), don't call this method.
        It will then be called automatically by the Env.
        """
        self.shouldStop = False
        self.entities = [
            Entity(self) for _ in range(FinBaseSimulation.initial_num_entities)
        ]
        for e in self.entities:
            r = Resource(asset_type="eur", number=100)
            e.resources["cash"] = r

    def stop(self):
        """Signal the simulation to stop at the next chance in the run loop.
        """
        self.shouldStop = True


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
        return "Resource(asset_type={}, number={}, allow_negative={})".format(
            self.asset_type, self.number, self.allow_negative
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
                "Resource {} must not become negative (when subtracting {} from {})".format(
                    self.id, number, self.number
                )
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
                "Asset type {} of Resource ".format(res.asset_type)
                + "{} cannot be put into Resource {} ".format(res.id, self.id)
                + "of asset type {}. Implicit ".format(self.asset_type)
                + "Recource conversions are not allowed."
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
        self.relations: Dict[Relation] = {}
        self.active = True

    def __repr__(self):
        return "Entity({})".format(self.id)

    def entity_relations(self, entity: "Entity"):
        return {k: r for k, r in self.relations.items() if r.entity == entity}

    def relations_of_role(self, role_name: str):
        return {
            k: r for k, r in self.relations.items() if r.role.name == role_name
        }

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

    @make_step(
        observation_space=spaces.Tuple(
            (
                spaces.Box(
                    low=np.array(
                        [-np.inf] * 2 * len(ASSET_TYPES) + [0] * 2 * len(ROLES)
                    ),
                    high=np.array(
                        [np.inf] * 2 * (len(ASSET_TYPES) + len(ROLES))
                    ),
                ),
                spaces.Discrete(len(REQUEST_TYPES)),
                spaces.Discrete(len(ROLES)),
            )
        ),
        observation_space_mapping=lambda self, request_type, requesting_entity, request_contents: (
            (
                self.resources_to_obs()
                + requesting_entity.resources_to_obs()
                + self.relations_to_obs()
                + requesting_entity.relations_to_obs()
            ),
            REQUEST_TYPES.index(request_type),
            ROLES.keys().index(request_contents["role_name"])
            if "role_name" in request_contents
            else 0,
        ),
        action_space=spaces.Discrete(2),
        action_space_mapping={0: None, 1: DeniedError},
        reward_mapping=lambda self, request_type, requesting_entity, request_contents: sum(
            self.resources_to_obs()
        ),
    )
    def check_request(
        self,
        request_type: str,
        requesting_entity: "Entity",
        request_contents: Dict,
    ):
        if request_type == "relation":
            role_name = request_contents["role_name"]
            if requesting_entity in [
                r.entity
                for _, r in self.relations.items()
                if r.role.name == role_name
            ]:
                raise DeniedError(
                    "{}-role relation already exists ".format(role_name)
                    + "between {} and {}".format(self, requesting_entity)
                )
            # test example
            elif role_name == "client" and len(self.resources) > 10:
                raise DeniedError("We don't take any more clients")

    def request_relation(
        self, role_name: str, requesting_entity: "Entity", key_to_use: str
    ):
        self.check_request(
            "relation", requesting_entity, {"role_name": role_name}
        )
        # TODO: Maybe put relation logic/inference code outside of entities
        role = ROLES[role_name]
        # Add my own inverse Relation back towards requester
        if role.inverse:
            if key_to_use in self.relations:
                raise ValueError(
                    "key_to_use {} already exists ".format(key_to_use)
                    + "in self.resources of {}".format(self)
                )
            inv_role = ROLES[role.inverse]
            back_rel = Relation(inv_role, requesting_entity)
            self.relations[key_to_use] = back_rel

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
