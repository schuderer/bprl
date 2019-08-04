"""Tests for the gym_fin.envs.pension_env module"""

# Stdlib imports
import logging
from math import floor
from unittest import mock

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


class MockClient:
    pass


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
    return env, company, pension_env.Client.maybe_make_client(env, force=True)


@pytest.fixture
def mockclient():
    return MockClient()


###############################################
# Seq
###############################################


def test_seq():
    class Something(pension_env.Seq):
        pass

    assert Something.create_id() == Something.create_id() - 1


###############################################
# Client
###############################################


def test_client_init(env):
    _ = pension_env.Client(env)


def test_client_factory_empty_env(env):
    with pytest.raises(pension_env.ClientError, match="no companies"):
        _ = pension_env.Client.maybe_make_client(env, force=True)


def test_client_factory_non_empty_env(mocker, env_company):
    mocker.patch("gym_fin.envs.pension_env.InsuranceCompany.create_membership")
    env, company = env_company
    client = pension_env.Client.maybe_make_client(env, force=True)
    company.create_membership.assert_called_once_with(client)
    assert client.pension_fund == company


def test_client_factory_no_client_because_bad_reputation(mocker, env_company):
    mocker.patch("gym_fin.envs.pension_env.InsuranceCompany.create_membership")
    env, company = env_company
    company.reputation = -9999999999
    client = pension_env.Client.maybe_make_client(env)
    assert client is None


def test_client_already_client(mocker, env_company_client):
    env, company, client = env_company_client
    with pytest.raises(pension_env.ClientError, match="already client"):
        client._become_client_of(company)


def test_client_give_or_take_positive(mocker, env_company_client):
    env, company, client = env_company_client
    initial_funds = client.funds
    success = client.give_or_take(123)
    assert success
    assert client.last_transaction == 123
    assert client.funds == initial_funds + 123


def test_client_give_or_take_negative(mocker, env_company_client):
    env, company, client = env_company_client
    initial_funds = client.funds
    success = client.give_or_take(-123)
    assert success
    assert client.last_transaction == -123
    assert client.funds == initial_funds - 123


def test_client_give_or_take_zero_funds_positive(mocker, env_company_client):
    env, company, client = env_company_client
    client.funds = 0
    success = client.give_or_take(1234)
    assert success
    assert client.last_transaction == 1234
    assert client.funds == 1234


def test_client_give_or_take_zero_funds_negative(mocker, env_company_client):
    env, company, client = env_company_client
    client.funds = 0
    success = client.give_or_take(-1234)
    assert not success
    assert client.last_transaction != -1234
    assert client.funds == 0


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
    assert client.give_or_take(+client.expectation)
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
    assert client.give_or_take(-client.income)
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_young_more_income(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age - 1
    # Doubling client's income
    assert client.give_or_take(+client.income)
    client.live_one_year()
    assert client.happiness >= initial_happiness


def test_client_live_one_year_young_no_funds(mocker, env_company_client):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age - 1
    client.funds = 0
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_young_no_funds_no_income(
    mocker, env_company_client
):
    env, company, client = env_company_client
    initial_happiness = client.happiness
    client.age = pension_env.Client.income_end_age - 1
    client.funds = -client.income
    client.live_one_year()
    assert client.happiness < initial_happiness


def test_client_live_one_year_recover_negative_happiness(
    mocker, env_company_client
):
    env, company, client = env_company_client
    client.age = pension_env.Client.income_end_age - 11
    client.happiness = -50
    for y in range(10):
        previous_happiness = client.happiness
        client.live_one_year()
        assert client.happiness > previous_happiness or client.happiness == 0


###############################################
# InsuranceCompany
###############################################


def test_insurance_company_init():
    _ = pension_env.InsuranceCompany()


def test_insurance_company_create_membership(mockclient, env_company):
    env, company = env_company
    company.create_membership(mockclient)
    assert mockclient in company.clients


def test_insurance_company_run_company_running_cost():
    company = pension_env.InsuranceCompany(investing=False)
    running_cost = pension_env.InsuranceCompany.running_cost
    initial_funds = company.funds
    company.run_company()
    assert company.funds == initial_funds - running_cost


def test_insurance_company_run_company_investing(env_company):
    env, company = env_company
    running_cost = pension_env.InsuranceCompany.running_cost
    initial_funds = company.funds
    # It's very certain that there will be profits after 50 years. :)
    for y in range(50):
        company.run_company()
        company.funds += running_cost
    assert company.funds > initial_funds - 50 * running_cost


def test_insurance_company_run_company_reputation(env_company):
    env, company = env_company
    company.reputation = 100
    initial_reputation = company.reputation
    company.run_company()
    assert company.reputation <= initial_reputation

    company.reputation = -100
    initial_reputation = company.reputation
    company.run_company()
    assert company.reputation > initial_reputation


def test_insurance_company_damage_reputation(env_company):
    env, company = env_company
    recovery = pension_env.InsuranceCompany.reputation_recovery
    initial_reputation = company.reputation
    company.damage_reputation(-100)
    company.run_company()
    assert company.reputation == initial_reputation - 100 + recovery

    company.reputation = 0
    initial_reputation = company.reputation
    with pytest.raises(
        pension_env.InsuranceCompanyError, match="must be negative"
    ):
        company.damage_reputation(100)


def test_insurance_company_do_debit_premium(env_company_client):
    env, company, client = env_company_client
    initial_funds = company.funds
    success = company.do_debit_premium(1234, client)
    assert success
    assert company.funds == initial_funds + 1234

    client.funds = 0
    success = company.do_debit_premium(1234, client)
    assert not success

    client.funds = 10000
    initial_funds = company.funds
    with pytest.raises(
        pension_env.InsuranceCompanyError, match="must be positive"
    ):
        company.do_debit_premium(-1234, client)


def test_insurance_company_do_pay_out(env_company_client):
    env, company, client = env_company_client
    initial_funds = company.funds
    success = company.do_pay_out(1234, client)
    assert success
    assert company.funds == initial_funds - 1234

    company.funds = 0
    success = company.do_pay_out(1234, client)
    assert not success

    company.funds = 10000
    initial_funds = company.funds
    with pytest.raises(
        pension_env.InsuranceCompanyError, match="must be positive"
    ):
        company.do_pay_out(-1234, client)


###############################################
# PensionEnv
###############################################


def test_pension_env_init():
    assert pension_env.PensionEnv()


def test_pension_env_companies():
    env = pension_env.PensionEnv()
    assert len(env.companies) == 0
    env.reset()
    assert len(env.companies) > 0


def test_pension_env_seed():
    env = pension_env.PensionEnv()
    env.seed(0)
    r1 = pension_env.np_random.choice(range(1000))
    env.seed(0)
    r2 = pension_env.np_random.choice(range(1000))
    assert r1 == r2


def test_pension_env_reset():
    env = pension_env.PensionEnv()
    env.seed(0)  # PensionEnv start state is currently deterministic, but still
    s1 = env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())
    env.seed(0)
    s2 = env.reset()
    assert (s1 == s2).all()


def test_pension_env_render():
    env = pension_env.PensionEnv()
    env.render()


def test_pension_env_close():
    env = pension_env.PensionEnv()
    env.close()


def test_pension_env_step_before_reset():
    env = pension_env.PensionEnv()
    with pytest.raises(pension_env.PensionEnvError, match="before reset"):
        env.step(env.action_space.sample())


@mock.patch.object(
    pension_env.InsuranceCompany, "do_debit_premium", return_value=True
)
def test_pension_env_step_debit_action(do_debit_premium):
    env = pension_env.PensionEnv()
    env.reset()
    debit_action = 0
    env.step(debit_action)
    do_debit_premium.assert_called_once()


@mock.patch.object(
    pension_env.InsuranceCompany, "do_pay_out", return_value=True
)
def test_pension_env_step_payout_action(do_pay_out):
    env = pension_env.PensionEnv()
    env.reset()
    payout_action = 1
    env.step(payout_action)
    do_pay_out.assert_called_once()


# @mock.patch.object(pension_env.Client, '_leave_cdf', return_value=1.0)
# @mock.patch.object(pension_env.Client, '_death_cdf', return_value=0.0)
# @mock.patch.object(pension_env.Client, '_new_cdf', return_value=0.0)
# @mock.patch.object(pension_env.Client, 'live_one_year')
# @mock.patch.object(pension_env.InsuranceCompany, 'run_company')
def test_pension_env_step_new_years(
    mocker
):  # _leave_cdf, _death_cdf, _new_cdf, live_one_year, run_company):
    """Given N Clients, env.year will advance with every Nth step(),
    InsuranceCompany.run_company will be called once per year and
    Client.live_one_year will be called once for each step.
    """
    # Patch Client object to never leave nor join nor die
    mocker.patch.object(pension_env.Client, "_leave_cdf", return_value=1.0)
    mocker.patch.object(pension_env.Client, "_death_cdf", return_value=0.0)
    mocker.patch.object(pension_env.Client, "_new_cdf", return_value=0.0)

    run_company = mocker.patch.object(
        pension_env.InsuranceCompany, "run_company"
    )

    env = pension_env.PensionEnv()
    env.reset()
    first_client = env.humans[0]
    second_client = pension_env.Client.maybe_make_client(env, force=True)
    env.humans.append(second_client)
    num_clients = len(env.humans)  # There are 2 clients
    print("num_clients", num_clients)
    mocker.spy(first_client, "live_one_year")
    mocker.spy(second_client, "live_one_year")
    steps = 10
    y0 = env.year  # probably 0
    debit_action = 0
    for _ in range(steps):
        env.step(debit_action)
    y5 = env.year
    expected_year_changes = floor(steps / num_clients)
    #       0 + floor(10 / 2)         ==  5
    assert y0 + expected_year_changes == y5
    assert run_company.call_count == y0 + expected_year_changes
    assert first_client.live_one_year.call_count == steps / num_clients
    assert second_client.live_one_year.call_count == steps / num_clients
