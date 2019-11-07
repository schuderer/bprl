import abc


class SimulationInterface(abc.ABC):
    """Interface for a subclass tasked with orchestrating the simulation.
    This class should be purely subject-matter-focused and not concern itself
    with observations, rewards, or actions. Those are dealt with by the
    environment-generating functionality (make_step(), get_env(), etc.)
    """

    @abc.abstractmethod
    def run(self):
        """The main control flow of the simulation. Only called by the agent
        if it is using callbacks directly.

        Terminates when an episode is over (does not need to terminate for
        infinite episodes). Use the environment's `reset` method to
        abort a possibly running episode and start a new episode.
        Use `stop` to just abort an episode without restarting.

        When using generate_env(), don't call this method from your code.
        It will be called automatically by the generated Env.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        """Return simulation to initial state.
        NOTE: This is NOT the Environment's reset() function.
        Only this function if your agent is using callbacks directly.

        When using generate_env(), don't call this method from your code.
        It will be called automatically by the generated Env.

        Generated Envs will call `reset` before calling `run`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self):
        """Tell the simulation to stop/abort the run. The recommended
        implementation is to set a `self.shouldStop` property to `True`.
        The subclass's implementation of `run` should check
        `self.shouldStop` in its loop and terminate the simulation if
        necessary.

        Please note that when resetting an environment, only `reset` will
        be called. So if necessary, make sure that your implementation of
        `reset` stops the simulation cleanly.

        When using generate_env(), don't call this method from your code.
        It will be called automatically by the generated Env.
        """
        raise NotImplementedError()
