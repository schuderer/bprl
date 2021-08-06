"""Define an example environment for the pension model & simulation"""

import numpy as np
from gym import spaces

from gym_fin.envs.sim_env import generate_env, make_step
import examples.pension as p


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


def reward_from_company(company, individual):
    if company.resources["cash"].number > 0:
        return 1
    else:
        return 0


def register_step_function(simulation_obj):
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
        reward_mapping=reward_from_company,
        simulation_obj=simulation_obj,
    )(p.PensionInsuranceCompany.determine_client_transaction)


def get_env_cls(max_individuals=50, max_steps=10):
    sim = p.PensionSim(max_individuals=max_individuals, max_steps=max_steps)
    register_step_function(sim)
    env_cls = generate_env(sim, "examples.pension.PensionInsuranceCompany.determine_client_transaction")
    return env_cls
