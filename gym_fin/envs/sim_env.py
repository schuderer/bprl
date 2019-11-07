from inspect import isclass
import logging
from threading import Event, Thread

# Todo: make it work with gym's "make" command
import gym
from gym import spaces

from gym_fin.envs.simulation_interface import SimulationInterface

logger = logging.getLogger(__name__)

env_metadata = {
    "step_views": {},
}


# maybe later: several step() interfacing points are possible, so provide
# options to consolidate the actions into one complete action
# space, but also the observations and steps (e.g. collect a number of requests
# before the "real" action step, and have them all as part of
# the "main" step()); and/or make it possible to attach independent agents
# to each of the decorated function steps()
def make_step(
    observation_space,
    observation_space_mapping,
    action_space,
    action_space_mapping,
    reward_mapping,
):
    """Decorator to turn an arbitrary function/method into a step() interface,
    providing what is essentially a hook/callback into agent code.
    Think of relevant functions as "decision functions", i.e. functions
    whose return value represents an action to take, whether something
    should be done/not done, etc.

    Params:
        - observation_space: AI Gym observation space definition
        - observation_space_mapping (callable): maps function
          parameters to observation space(s)
        - action_space: AI Gym action space definition
        - action_space_mapping (dict or callable): maps return values or thrown
          exceptions to action indices or action values in the action_space
        - reward_mapping (callable): function that returns a reward

    Any callables will be called with the parameters of the decorated function.
    """

    assert callable(
        observation_space_mapping
    ), "observation_space_mapping must be a function reference"
    obs_from_args = observation_space_mapping

    def perform_action(action):
        if callable(action_space_mapping):
            return action_space_mapping(action)
        else:
            action_val = action_space_mapping[action]
            if isclass(action_val) and issubclass(action_val, Exception):
                raise action_val("action {}".format(action))
            else:
                return action_val

    if not callable(action_space_mapping):
        assert type(action_space) == spaces.Discrete, (
            "action_space_mapping dict requires "
            "action_space to be of type Discrete"
        )
        assert action_space.n == len(action_space_mapping), (
            "Mismatch in number of actions between "
            "action_space_mapping and action_space"
        )

    def decorator(func):
        module = __name__
        name = "{}.{}".format(module, func.__qualname__)

        # pre-register step_view
        global env_metadata
        if name in env_metadata["step_views"]:
            raise RuntimeError("Function {} registered twice.".format(name))
        else:
            env_metadata["step_views"][name] = {
                "observation_space": observation_space,
                "action_space": action_space,
                "callback": None,  # todo: anything else?
            }

        def step_wrapper(*args, **kwargs):
            """This is the "decision function" called by simulation
            code which is providing an AI-Gym-step()-like interface.
            """
            global env_metadata
            step_view = env_metadata["step_views"][name]
            if step_view["callback"]:
                before_obs = obs_from_args(*args, **kwargs)
                before_reward = reward_mapping(*args, **kwargs)
                before_info = {}
                action = step_view["callback"](
                    before_obs, before_reward, before_info
                )
                return perform_action(action)
            else:
                # No subscribers to step(), execute func's code instead
                return func(*args, **kwargs)

        return step_wrapper

    return decorator


def register_env_callback(name: str, callback):  # , flatten_obs=True):
    """Register a callable as the `callback` to be called from the
    function referred to by the fully qualified `name`.

    Params:
        - name (str): The fully qualified name of the Entity's function you
          want to replace by an agent, e.g. `"fin_entity.Entity.check_request"`
        - callback: The agent's function to register.
          It will be passed the parameters `obs`, `reward` and
          `info`. It should return an `action` value.
        # - flatten_obs (bool): Should the observation be passed to `callback`
        #   as a `dict`. If False, its first level is flattened into a
        #   positional vector. True by default.
    """
    global env_metadata
    step_views = env_metadata["step_views"]
    if name in step_views:
        step_view = step_views[name]
        if "callback" in step_view and step_view["callback"]:
            logger.warn(
                "Overwriting existing callback for {}".format(name)
            )  # todo log
        step_view.update(
            {
                "callback": callback,
                # "flatten_obs": flatten_obs,
            }
        )
    else:
        raise ValueError(
            "No such registered function: '{}'. Existing functions are: {}"
        ).format(name, list(step_views.keys()))


# https://stackoverflow.com/a/18506625/2654383
def generate_env(world: SimulationInterface, name: str):
    """Generates and returns an OpenAI Gym compatible Environment class.

    Params:
        - world: A simulation object which implements the methods
          `reset` and `run` of `SimulationInterface`.
        - name: A string describing a method or function inside `world` or any
          objects therein. This method or function is only available if it has
          been turned into a `step` callback using the `@make_step` decorator.
    """

    class CallbackEnv(gym.Env):
        # Asynchronicity management code
        def run_user_code(self):
            self.user_done.clear()
            self.sim_done.set()
            logger.debug("callback waiting for turn...")
            self.user_done.wait()

        def run_simulation_code(self):
            self.sim_done.clear()
            self.user_done.set()
            logger.debug("user code waiting for turn...")
            self.sim_done.wait()

        def simulation_async(self, simulation_obj):
            logger.debug("waiting to start simulation")
            self.user_done.wait()
            self.sim_done.clear()
            logger.debug("running simulation...")
            simulation_obj.reset()
            simulation_obj.run()
            logger.debug("simulation ended")
            self.sim_done.set()

        # Environment proper
        metadata = {"render.modes": ["human", "ascii"]}  # todo

        def __init__(self):
            self.viewer = None
            self.obs = None
            self.reward = None
            self.done = False
            self.info = None
            self.action = None
            self.simulation_thread = None
            self.simulation = world
            self.user_done = Event()
            self.sim_done = Event()
            self.user_done.clear()
            self.sim_done.set()

            def _my_callback(obs, reward, info):
                self.obs = obs
                self.reward = reward
                self.info = info
                logger.debug(
                    "callback observation={} releasing turn".format(obs)
                )
                self.run_user_code()
                # We know the action now
                logger.debug(
                    "callback {}: got turn, returning action {}".format(
                        obs, self.action
                    )
                )
                return self.action

            register_env_callback(name, _my_callback)
            global env_metadata
            step_views = env_metadata["step_views"]
            self.observation_space = step_views[name]["observation_space"]
            self.action_space = step_views[name]["action_space"]

        def reset(self):
            """Resets the state of the environment, returning
            an initial observation.

            Returns:
                observation: the initial observation of the space.
                (Initial reward is assumed to be 0.)
            """
            logger.debug("Starting simulation thread")
            self.simulation_thread = Thread(
                target=self.simulation_async, args=(self.simulation,)
            )
            self.simulation_thread.start()

            logger.debug("reset: releasing turn")
            self.run_simulation_code()
            # We know the observation now
            logger.debug("reset got turn, returning obs {}".format(self.obs))
            return self.obs  # is set because callback has been called

        def step(self, action):
            """Run one timestep of the environment's dynamics. When end of
            the episode is reached, reset() should be called to reset the
            environment's internal state.

            Args:
                action: an action provided by the environment

            Returns:
                - (observation, reward, done, info)
                - observation: agent's observation of the current environment
                - reward: float, amount of reward due to the previous action
                - done: a boolean, indicating whether the episode has ended
                - info: a dictionary containing other diagnostic information
                        from the previous action
            """
            self.action = action
            logger.debug("step: action={} releasing turn".format(action))
            self.run_simulation_code()
            # We know the observation, reward, info now
            self.done = not self.simulation_thread.is_alive()
            logger.debug(
                "step: got turn, returning (obs={}, done={})".format(
                    self.obs, self.done
                )
            )
            return (
                self.obs,
                self.reward,
                self.done,
                self.info,
            )  # properties are updated because callback has been called

        def stop(self):
            # todo
            pass

        def render(self, mode="human", close=False):
            pass
            # raise NotImplementedError

        def close(self):
            if self.viewer:
                self.viewer.close()
                self.viewer = None

    return CallbackEnv
