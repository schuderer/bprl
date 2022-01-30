"""Define an example environment for the pension model & simulation"""

import numpy as np
from gym import spaces

from gym_fin.envs.sim_env import generate_env, make_step
import examples.pension as p


# def obs_from_company(company, individual):
#     # [Curr Client Age, Company funds, Reputation, Num Clients]
#     client_age_obs = min(individual.age, 100) / 100
#     funds_obs = min(company.resources["cash"].number, 1000000) / 1000000
#     public_opinion = company.world.find_entities(p.PublicOpinion)[0]
#     reputation_obs = (max(public_opinion.reputation(company), -10000) / 10000) + 1.0
#     clients = [
#         contract.entities[1]
#         for contract in company.find_contracts(type="insurance")
#         if contract.entities[1].active
#     ]
#     clients_obs = min(len(clients), 1000) / 1000
#
#     return np.array([
#         client_age_obs,
#         funds_obs,
#         reputation_obs,
#         clients_obs,
#     ])

def obs_from_company(company, individual):
    # [Curr Client Age, Company funds, Reputation, Num Clients]
    client_age_obs = np.clip(individual.age, 0, 100)  # / 100
    funds_obs = np.clip(company.resources["cash"].number, 0, 1000000)  # / 1000000
    public_opinion = company.world.find_entities(p.PublicOpinion)[0]
    reputation_obs = np.clip(public_opinion.reputation(company), -10000, 0)  # (max(public_opinion.reputation(company), -10000) / 10000) + 1.0
    clients = [
        contract.entities[1]
        for contract in company.find_contracts(type="insurance")
        if contract.entities[1].active
    ]
    clients_obs = np.clip(len(clients), 0, 1000)  # / 1000

    obs = np.array([
        client_age_obs,
        funds_obs,
        reputation_obs,
        clients_obs,
    ])
    # print(obs)
    return obs


def reward_from_company(company, individual):
    rew = 1 if company.resources["cash"].number > 0 else 0
    # print(f"reward: {rew}")
    return rew


def register_step_function():
    def temp_act_map(a):
        # a_out = np.clip(float(a) * 100000 - 50000, -50000, 50000)  # min(max(0, w * 1000), 1000),
        a_out = np.clip(float(a) * 100000 - 50000, -50000, 50000)  # min(max(0, w * 1000), 1000),
        # print(f"converted {a} to {a_out}")
        # return np.array(float(input("Take (neg) / give (pos) amount: ")))
        return a_out

    p.PensionInsuranceCompany.determine_client_transaction = make_step(
        # [Curr Client Age, Company funds, Reputation, Num Clients]
        # (normalized to make broadly usable)
        observation_space=spaces.Box(
            # low=np.array([0.0] * 4),
            # high=np.array([1.0] * 4),
            # obs boxes: curr_client_age, funds, reput, num_clients
            low=np.array([0, 0, -10000, 0]),
            high=np.array([100, 1000000, 0, 1000]),
            dtype=np.float32
        ),
        observation_space_mapping=obs_from_company,
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        action_space_mapping=temp_act_map,
        # action_space=spaces.Box(low=-50000, high=50000, shape=(1,)),
        # action_space_mapping=lambda a: a,  # temp_act_map,
        reward_mapping=reward_from_company,
    )(p.PensionInsuranceCompany.determine_client_transaction)


def get_env_cls(max_individuals=50, max_days=365 * 300):
    sim = p.PensionSim(max_individuals=max_individuals, max_days=max_days)
    register_step_function()
    env_cls = generate_env(sim, "examples.pension.PensionInsuranceCompany.determine_client_transaction")
    return env_cls
