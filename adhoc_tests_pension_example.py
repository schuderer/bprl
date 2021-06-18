# Currently only contains a few manual tests of the simulation code
import logging
import sys

from importlib import reload

import examples.pension as p

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logging.setLevel(logging.INFO)
logging.getLogger("gym_fin.envs.fin_base_sim").setLevel(logging.DEBUG)
logging.getLogger("examples.pension").setLevel(logging.DEBUG)


def notest_pensionsim():
    reload(p)
    s = p.PensionSim()

    s.reset()
    s.run()


def notest_pensioninsurancecompany():
    reload(p)
    s = p.PensionSim()
    s.reset()
    c = p.PensionInsuranceCompany(s)
    c.perform_increment()
    print(c)


def notest_individual():
    reload(p)
    s = p.PensionSim()
    s.reset()
    i = p.Individual(s)
    s.entities.append(i)
    for _ in range(10):
        # i.resources["cash"].take(10000)
        if i.age > 26:
            i.living_expenses = 50000
        s.run_increment()
    po = s.find_entities(p.PublicOpinion)[0]
    po.perform_increment()


def test_pensionsim_static():
    reload(p)
    s = p.PensionSim(max_individuals=10)
    s.reset()
    s.run()  # max_t=20 * 365)


def notest_pensionsim_fixed_agent():
    import numpy as np
    from gym import spaces
    from gym.envs.registration import register
    from gym_fin.envs.sim_env import generate_env, make_step
    import examples.pension as p
    reload(p)

    def obs_from_company(company, individual):
        # [Curr Client Age, Company funds, Reputation, Num Clients]
        client_age_obs = min(individual.age, 100) / 100
        funds_obs = min(company.resources["cash"].number, 1000000) / 1000000
        public_opinion = company.world.find_entities(p.PublicOpinion)[0]
        reputation_obs = (max(public_opinion.reputation(company), -10000) / 10000) + 1.0
        clients = [
            contract.entities[1]
            for contract in company.find_contracts(type="insurance")
            if contract.entities[1].active
        ]
        clients_obs = min(len(clients), 1000) / 1000

        return np.array([
            client_age_obs,
            funds_obs,
            reputation_obs,
            clients_obs,
        ])

    def reward_from_company(c):
        return None

    p.PensionInsuranceCompany.determine_client_transaction = make_step(
        # [Curr Client Age, Company funds, Reputation, Num Clients]
        # (normalized to make broadly usable)
        observation_space=spaces.Box(
            low=np.array([0.0] * 4),
            high=np.array([1.0] * 4)
        ),
        observation_space_mapping=obs_from_company,
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,)),
        action_space_mapping=lambda w: min(max(0, w * 1000), 1000),
        reward_mapping=reward_from_company
    )(p.PensionInsuranceCompany.determine_client_transaction)

    s = p.PensionSim(max_individuals=10)
    env_cls = generate_env(s, "examples.pension.PensionInsuranceCompany.determine_client_transaction")
    register(
        id="Greenhouse-v0",
        entry_point=env_cls,  # "env_def:env_cls",
    )

    s.reset()
    s.run()  # max_t=20 * 365)


if __name__ == "__main__":
    func_name: str
    for func_name in dir(sys.modules[__name__]):
        func = getattr(sys.modules[__name__], func_name)
        if callable(func) and func_name.startswith("test_"):
            print(f"Calling {func_name}:")
            func()
            print("###############################")
