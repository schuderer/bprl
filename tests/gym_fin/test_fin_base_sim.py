"""Tests for the gym_fin.envs.fin_base_sim module"""

# Stdlib imports
# import logging
from importlib import reload
from unittest import mock

# Third-party imports
# from gym.utils import seeding
import pytest

# Application level imports
import gym_fin.envs.fin_base_sim as f


@pytest.fixture
def sim():
    reload(f)
    s = f.FinBaseSimulation()
    return s


@pytest.fixture
def entity(sim):
    e = f.Entity(sim)
    return e


@pytest.fixture
def entities(sim):
    return f.Entity(sim), f.Entity(sim)


@pytest.fixture
def role(sim):
    f.Role("bla", "invbla", "blarel")
    return f.Role("bla")


@pytest.fixture
def resource():
    return f.Resource("usd", 42, True)


@pytest.fixture
def trade_entities(entities):
    e1, e2 = entities
    c1 = f.Trade(e1, "buy", 100, "eur", e2, 120, "usd", "somereference")
    c1.draft = True
    c2 = c1.get_inverse()
    e1.contracts.append(c1)
    e2.contracts.append(c2)
    return e1, e2


###############################################
# General stuff
###############################################


def test_denied_error():
    assert f.DeniedError("bla").counter_offer is None
    assert f.DeniedError("bla", 3).counter_offer == 3


def test_seq():
    class Something(f.Seq):
        def __init__(self):
            self.id = self.create_id()

    assert Something().id != Something().id


###############################################
# Role
###############################################


def test_role(sim):
    with pytest.raises(ValueError):
        f.Role("bla")
    assert "bla" not in f.Role.roles()
    f.Role("bla", "invbla", "blarel")
    assert "bla" in f.Role.roles()
    r = f.Role("bla")
    assert r.inverse == "invbla"
    print(r)


###############################################
# Contract
###############################################


def test_contract(entities, role: f.Role):
    e1, e2 = entities
    c = f.Contract(e1, role.role, e2, "ref")
    print(c)
    assert c.entities == [e1, e2]
    assert not c.is_fulfilled()
    assert c.to_contract_request()["my_role"] == role.role
    i = c.get_inverse()
    assert i is not c
    assert i.id != c.id
    assert i.roles[0] == c.roles[1]


def test_contract_deepcopy_regression(entities, role):
    """Should not create copy of entities when creating the inverse contract"""
    e1, e2 = entities
    c = f.Contract(e1, role.role, e2, "ref")
    i = c.get_inverse()
    assert i.entities[0] is e2
    assert i.entities[1] is e1


###############################################
# Trade
###############################################


def test_trade(entities, role: f.Role):
    e1, e2 = entities
    c = f.Trade(e1, role.role, 200, "eur", e2, 1, "AAPL", "ref")
    print(c)
    assert c.entities == [e1, e2]
    cr = c.to_contract_request()
    assert cr["asked_number"] == 1
    assert cr["offered_number"] == 200


###############################################
# FinBaseSimulation
###############################################


def test_finbasesimulation(sim):
    sim.reset()
    sim.run()


def test_finbasesimulation_run_before_reset(sim):
    with pytest.raises(RuntimeError):
        sim.run()


def test_finbasesimulation_stop(sim):
    sim.reset()
    sim.stop()
    assert sim.should_stop


def test_finbasesimulation_seed():
    reload(f)
    sim1 = f.FinBaseSimulation()
    sim1.seed(24)
    i1 = sim1.np_random.randint(100000000)
    sim2 = f.FinBaseSimulation()
    sim2.seed(24)
    i2 = sim2.np_random.randint(100000000)
    assert i1 == i2


def test_finbasesimulation_find_entities(sim):
    class Bla(f.Entity):
        pass

    class Blu(f.Entity):
        pass

    sim.entities = [Bla(sim), Bla(sim), Blu(sim)]
    sim.entities[2].active = False
    assert len(sim.find_entities(f.Entity)) == 2
    assert len(sim.find_entities(f.Entity, False)) == 1
    assert len(sim.find_entities(Bla, True)) == 2
    assert len(sim.find_entities(Bla, False)) == 0
    assert len(sim.find_entities(Blu)) == 0
    assert len(sim.find_entities(Blu, False)) == 1


###############################################
# Resource
###############################################


def test_resource():
    f.Resource("usd", 42)


def test_resource_unknown_asset_type():
    with pytest.raises(ValueError):
        f.Resource("saoenthaueosnth", 42)


def test_resource_take(resource):
    res1num = resource.number
    res2 = resource.take(3)
    assert res2.number == 3
    assert resource.number + res2.number == res1num
    assert res2.asset_type == resource.asset_type


def test_resource_put(resource):
    res1num = resource.number
    res2 = f.Resource("usd", 3)
    resource.put(res2)
    assert resource.number == res1num + 3
    assert res2.number == 0
    assert res2.asset_type == resource.asset_type


def test_resource_take_too_much():
    res = f.Resource("usd", 2, allow_negative=False)
    with pytest.raises(f.DeniedError):
        res.take(3)


def test_resource_put_wrong_type(resource):
    res2 = f.Resource("eur", 3)
    with pytest.raises(f.DeniedError):
        resource.put(res2)


###############################################
# Entity
###############################################


def test_entity(sim):
    f.Entity(sim)


def test_entity_ensure_active(entity):
    entity.ensure_active()
    entity.active = False
    with pytest.raises(f.EntityInactiveError):
        entity.ensure_active()


def test_entity_find_contracts(sim, entities, role):
    e1, e2 = entities
    c1 = f.Contract(e1, role.role, e2, "ref")
    c2 = f.Contract(e1, role.role, e2, "NOMATCH")
    c3 = f.Contract(e1, role.role, f.Entity(e1.world), "ref")
    e1.contracts += [c1, c2, c3]
    assert e1.find_contracts(
        type=role.relation, other=e2, reference="ref"
    ) == [c1]
    assert e1.find_contracts(type=role.relation, other=e2) == [c1, c2]
    assert e1.find_contracts(type=role.relation) == [c1, c2, c3]


@mock.patch("{}.Entity.check_BOGUS_request".format(f.__name__), create=True)
def test_entity_check_request(mock_check_func, entities):
    e1, e2 = entities
    e1.check_request("BOGUS", e2, {})
    mock_check_func.assert_called_once_with("BOGUS", e2, {})


def test_entity_intransfer_request(trade_entities):
    me, other = trade_entities
    other.check_intransfer_request("trade", me, {})


def test_entity_outtransfer_request(trade_entities):
    me, other = trade_entities
    other.check_outtransfer_request(
        "trade",
        me,
        {
            "reference": me.contracts[0].reference,
            "number": me.contracts[0].numbers[1],
            "asset_type": me.contracts[0].assets[1],
        },
    )
    # partial requests are possible:
    other.check_outtransfer_request(
        "trade",
        me,
        {
            "reference": me.contracts[0].reference,
            "number": me.contracts[0].numbers[1] / 2,
            "asset_type": me.contracts[0].assets[1],
        },
    )


def test_entity_outtransfer_request_deny_no_contract(entities):
    me, other = entities
    with pytest.raises(f.DeniedError, match="matching contract"):
        other.check_outtransfer_request(
            "trade",
            me,
            {
                "reference": "referencedoesntmatter",
                "number": 3,
                "asset_type": "doesntmatter",
            },
        )


def test_entity_outtransfer_request_deny_fulfilled(trade_entities):
    me, other = trade_entities
    other.contracts[0].fulfilled[0] = True
    with pytest.raises(f.DeniedError, match="fulfilled"):
        other.check_outtransfer_request(
            "trade",
            me,
            {
                "reference": me.contracts[0].reference,
                "number": me.contracts[0].numbers[1],
                "asset_type": me.contracts[0].assets[1],
            },
        )


def test_entity_outtransfer_request_deny_too_much(trade_entities):
    me, other = trade_entities
    with pytest.raises(f.DeniedError, match="only"):
        other.check_outtransfer_request(
            "trade",
            me,
            {
                "reference": me.contracts[0].reference,
                "number": me.contracts[0].numbers[1] + 1,
                "asset_type": me.contracts[0].assets[1],
            },
        )


def test_entity_outtransfer_request_deny_wrong_asset(trade_entities):
    me, other = trade_entities
    with pytest.raises(f.DeniedError, match="asset"):
        other.check_outtransfer_request(
            "trade",
            me,
            {
                "reference": me.contracts[0].reference,
                "number": me.contracts[0].numbers[1],
                "asset_type": "bogus_asset",
            },
        )


def test_entity_outtransfer_request_error_more_than_1(trade_entities):
    me, other = trade_entities
    other.contracts.append(other.contracts[0])
    with pytest.raises(ValueError):
        other.check_outtransfer_request(
            "trade",
            me,
            {
                "reference": me.contracts[0].reference,
                "number": me.contracts[0].numbers[1],
                "asset_type": me.contracts[0].assets[1],
            },
        )


def test_entity_request_transfer(trade_entities):
    me, other = trade_entities
    res = f.Resource(
        me.contracts[0].assets[1],
        me.contracts[0].numbers[1],
        allow_negative=False,
    )
    other.resources[me.contracts[0].reference] = res
    r1 = other.request_transfer(
        number=me.contracts[0].numbers[1] / 2,
        asset_type=me.contracts[0].assets[1],
        reference=me.contracts[0].reference,
        requesting_entity=me,
    )
    r2 = other.request_transfer(
        number=me.contracts[0].numbers[1] / 2,
        asset_type=me.contracts[0].assets[1],
        reference=me.contracts[0].reference,
        requesting_entity=me,
    )
    assert other.resources[me.contracts[0].reference].number == 0
    assert r1.number == me.contracts[0].numbers[1] / 2
    assert r2.number == me.contracts[0].numbers[1] / 2


def test_entity_request_transfer_number_higher_than_contract(trade_entities):
    me, other = trade_entities
    res = f.Resource(
        me.contracts[0].assets[1],
        me.contracts[0].numbers[1] + 1,
        allow_negative=False,
    )
    other.resources[me.contracts[0].reference] = res
    with pytest.raises(f.DeniedError, match="only"):
        other.request_transfer(
            number=me.contracts[0].numbers[1] + 1,
            asset_type=me.contracts[0].assets[1],
            reference=me.contracts[0].reference,
            requesting_entity=me,
        )


def test_entity_request_transfer_insufficient_funds(trade_entities):
    me, other = trade_entities
    res = f.Resource(
        me.contracts[0].assets[1],
        me.contracts[0].numbers[1] - 1,
        allow_negative=False,
    )
    other.resources[me.contracts[0].reference] = res
    with pytest.raises(f.DeniedError, match="must not become negative"):
        other.request_transfer(
            number=me.contracts[0].numbers[1],
            asset_type=me.contracts[0].assets[1],
            reference=me.contracts[0].reference,
            requesting_entity=me,
        )


def test_entity_receive_transfer(trade_entities):
    me, other = trade_entities
    res = f.Resource(
        me.contracts[0].assets[0],
        me.contracts[0].numbers[0],
        allow_negative=False,
    )
    other.receive_transfer(
        resource=res, reference=me.contracts[0].reference, requesting_entity=me
    )
    assert (
        other.resources[me.contracts[0].reference].number
        == me.contracts[0].numbers[0]
    )


@mock.patch(
    "{}.Entity.check_blarel_contract_request".format(f.__name__), create=True
)
def test_entity_request_contract(mock_check_func, entities, role: f.Role):
    me, other = entities
    c = f.Contract(me, role.role, other, "blaref")
    other.request_contract(c)
    mock_check_func.assert_called_once()
    other_c = other.find_contracts(role.relation, me, "blaref")
    assert len(other_c) == 1


@mock.patch(
    "{}.Entity.check_blarel_contract_request".format(f.__name__), create=True
)
def test_entity_request_contract_not_draft(
    mock_check_func, entities, role: f.Role
):
    me, other = entities
    c = f.Contract(me, role.role, other, "blaref")
    c.draft = False
    with pytest.raises(f.DeniedError, match="draft"):
        other.request_contract(c)


@mock.patch(
    "{}.Entity.check_blarel_contract_request".format(f.__name__), create=True
)
def test_entity_request_contract_not_a_party(
    mock_check_func, entities, entity, role: f.Role
):
    me, other = entities
    someone_else = entity
    c = f.Contract(me, role.role, someone_else, "blaref")
    with pytest.raises(f.DeniedError, match="must be a party"):
        other.request_contract(c)
