"""Tests for the gym_fin.envs.sim_env module"""

# stdlib imports
import importlib
import logging

# Third-party imports
import pytest

# Application-level imports
import gym_fin.envs.sim_env as sim


logging.basicConfig(level=logging.DEBUG)


def test_plugin_handlers():
    importlib.reload(sim)

    @sim.expose_to_plugins
    def add(x, y=3):
        return x + y

    called = {}

    def before(x, y=3):
        print(f"before, x={x}, y={y}")
        called["before"] = True

    sim.attach_handler(before, add, "before")

    assert "before" not in called
    assert add(3) == 6
    assert called["before"]
    assert add(3, 4) == 7

    def after(x, y=3, _result=0):
        print(f"after, x={x}, y={y}, r={_result}")
        called["after"] = True

    sim.attach_handler(
        after,
        "tests.gym_fin.test_sim_env.test_plugin_handlers.<locals>.add",
        "after",
    )

    assert "after" not in called
    assert add(2, 1) == 3
    assert called["after"]

    def instead(x, y=3):
        called["instead"] = True
        return x * y

    sim.attach_handler(instead, add, "instead")

    assert "instead" not in called
    assert add(3, 5) == 15
    assert add(3) == 9
    assert called["instead"]


def test_plugin_handler_signature_verification():
    importlib.reload(sim)

    @sim.expose_to_plugins
    def bla(a, b: int, c=42):
        pass

    def before1(x, b: int, c=42):
        pass

    with pytest.raises(AssertionError, match="signature"):
        sim.attach_handler(before1, bla, "before")

    def before2(a, b: str, c=42):
        pass

    with pytest.raises(AssertionError, match="signature"):
        sim.attach_handler(before2, bla, "before")

    def before3(a, b):
        pass

    with pytest.raises(AssertionError, match="signature"):
        sim.attach_handler(before3, bla, "before")

    def before4(a, b: int, c=27):
        pass

    with pytest.raises(AssertionError, match="signature"):
        sim.attach_handler(before4, bla, "before")

    def after(a, b: int, c=42):
        pass

    with pytest.raises(AssertionError, match="after"):
        sim.attach_handler(after, bla, "after")


def test_plugin_handler_change_args_and_result():
    importlib.reload(sim)

    @sim.expose_to_plugins
    def sub(x, y, swap=False):
        return y - x if swap else x - y

    called = {}

    def before_change(x, y, swap=False):
        print(f"before, x={x}, y={y}")
        called["before_change"] = True
        args = sim.ChangedArgs(x + 1, y - 1, swap=True)
        return args

    sim.attach_handler(before_change, sub, "before")

    assert "before_change" not in called
    assert sub(4, 3) == (3 - 1) - (4 + 1)
    assert "before_change" in called

    def after_change(x, y, swap=False, _result=None):
        print(f"after, x={x}, y={y}, _result={_result}")
        called["after_change"] = True
        args = sim.ChangedResult(_result - x * y)
        return args

    sim.attach_handler(after_change, sub, "after")

    assert "after_change" not in called
    assert sub(4, 3) == ((3 - 1) - (4 + 1)) - (4 + 1) * (3 - 1)
    assert "after_change" in called


def test_plugin_handler_iterables():
    importlib.reload(sim)

    def run_loop():
        counter = 0
        for i in sim.expose_to_plugins(range(10)):
            counter += 1
        return counter

    assert run_loop() == 10


def test_plugin_handler_iterables_before():
    importlib.reload(sim)

    def run_loop():
        counter = 0
        for i in sim.expose_to_plugins(range(10)):
            counter += 1
        return counter

    called = {"before_next": 0}

    def before_next(self):
        called["before_next"] += 1

    sim.attach_handler(before_next, "range(0, 10)", "before")

    assert run_loop() == 10
    assert called["before_next"] == 11  # 10 successful, 1 raises StopIteration

    # # Tests for changing iterator.__next__'s only argument "self",
    # # effectively swapping out the underlying iterator:
    # # DOES NOT WORK: TypeError: descriptor '__next__'
    # # requires a 'range_iterator' object but received a 'list_iterator'
    # new_iterator = iter([3, 1, 4])
    #
    # def before_next(self):
    #     print("Argument 'self' would have been", repr(self))
    #     res = sim.ChangedArgs(new_iterator)
    #     print("Now, 'self' will be", repr(new_iterator))
    #     return res
    #
    # sim.attach_handler(before_next, "range(0, 10)", "before")
    #
    # assert run_loop() == 3


def test_plugin_handler_iterables_after():
    importlib.reload(sim)

    my_dict = {"a": 3, "b": 2, "c": 1}

    def run_loop():
        counter = 0
        sum = 0
        for k, v in sim.expose_to_plugins(
            my_dict.items(), override_name="bla"
        ):
            counter += 1
            sum += v
        return (counter, sum)

    called = {"after_next": 0}

    def after_next(self, _result=None):
        called["after_next"] += 1

    sim.attach_handler(after_next, "bla", "after")

    count, sum = run_loop()
    assert count == 3
    assert sum == 3 + 2 + 1
    # 3 successful, 4th never reaches 'after' due to StopIteration:
    assert called["after_next"] == 3

    # Tests for changing the iterator's results:
    def after_next(self, _result=None):
        print("Result would have been", _result)
        res = sim.ChangedResult(("life", 42))
        print("Now, result will be", res)
        return res

    sim.attach_handler(after_next, "bla", "after")

    count, sum = run_loop()
    assert count == 3
    assert sum == 42 * 3


def test_plugin_handler_iterables_instead():
    importlib.reload(sim)

    def run_loop():
        counter = 0
        for i in sim.expose_to_plugins([2, 4, 6, 8], override_name="sth"):
            counter += 1
        return counter

    other_iterator = iter(range(10))

    def instead_next(self):
        return next(other_iterator)

    assert run_loop() == 4

    sim.attach_handler(instead_next, "sth", "instead")

    assert run_loop() == 10
