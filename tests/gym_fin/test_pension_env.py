"""Tests for the gym_fin.envs.pension_env module"""

# Stdlib imports
import logging

# Third-party imports
from gym.utils import seeding
import pytest

# from pytest_mock import mocker

# Application level imports
from gym_fin.envs import pension_env


class MockEnv:
    def __init__(self):
        self.year = 0
        self.companies = []
        self.humans = []


@pytest.fixture(autouse=True)
def seed():
    # TODO: ugly global -- refactor in code?
    pension_env.np_random, seed = seeding.np_random(0)
    return seed


@pytest.fixture(autouse=True)
def loglevel_debug():
    pension_env.logger.setLevel(logging.DEBUG)


@pytest.fixture
def env():
    return MockEnv()


@pytest.fixture
def env_company(env):
    company = pension_env.InsuranceCompany()
    env.companies.append(company)
    return env, company


@pytest.fixture
def env_company_client(env_company):
    env, company = env_company
    return env, company, pension_env.Client(env)


def test_seq():
    class Something(pension_env.Seq):
        pass

    assert Something.create_id() == Something.create_id() - 1


def test_client_init_empty_env(env):
    with pytest.raises(pension_env.ClientError, match="no companies"):
        _ = pension_env.Client(env)


def test_client_init_non_empty_env(mocker, env_company):
    mocker.patch("gym_fin.envs.pension_env.InsuranceCompany.create_membership")
    env, company = env_company
    client = pension_env.Client(env)
    company.create_membership.assert_called_once_with(client)
    assert client.pension_fund == company


def test_client_live_one_year_age(mocker, env_company_client):
    env, company, client = env_company_client
    initial_age = client.age
    client.live_one_year()
    # out, err = capsys.readouterr()
    assert client.age == initial_age + 1


def test_client_live_one_year_earn_income(mocker, env_company_client):
    env, company, client = env_company_client
    initial_funds = client.funds
    client.live_one_year()
    assert (
        client.funds == initial_funds + client.income - client.living_expenses
    )


def test_client_live_one_year_die(mocker, env_company_client, caplog):
    env, company, client = env_company_client
    client.age = 9999999999
    client.live_one_year()
    assert not client.active
    assert "RIP" in caplog.text


def test_client_live_one_year_leave(mocker, env_company_client, caplog):
    env, company, client = env_company_client
    initial_company_reputation = company.reputation
    client.happiness = -9999999999
    client.live_one_year()
    assert not client.active
    assert "Goodbye" in caplog.text
    assert company.reputation < initial_company_reputation


def test_client_live_one_year_old_no_pension(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age + 1
    # We won't give client income
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_old_pension(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age + 1
    print(client.funds, client.expectation)
    client.give_or_take(+client.expectation)
    client.live_one_year()
    assert client.happiness >= initial_happiness


def test_client_live_one_year_old_no_funds(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age + 1
    client.funds = 0
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_young_no_income(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age - 1
    # Taking client's income away from them
    client.give_or_take(-client.income)
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_young_more_income(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age - 1
    # Doubling client's income
    client.give_or_take(+client.income)
    client.live_one_year()
    assert client.happiness >= initial_happiness


def test_client_live_one_year_young_no_funds(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age - 1
    client.funds = 0
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_recover_negative_happiness(
    mocker, env_company_client
):
    env, company, client = env_company_client
    initial_happiness = -100
    client.happiness = initial_happiness
    client.live_one_year()
    assert client.happiness > initial_happiness
