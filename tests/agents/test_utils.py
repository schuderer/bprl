"""Tests for the agents.utils module"""

import pytest


class DidRun(RuntimeError):
    pass


def test_lazystr():
    from agents import utils

    def raise_did_run():
        raise DidRun()

    # creation should not execute function:
    ls = utils.LazyStr(raise_did_run)

    # string evaluation should execute function:
    with pytest.raises(DidRun):
        str(ls)
