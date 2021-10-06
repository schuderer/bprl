# Currently only contains a few manual tests of the simulation code
import logging
import sys
from importlib import reload

import numpy as np

import examples.pension as p

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logging.setLevel(logging.INFO)
logging.getLogger("gym_fin.envs.fin_base_sim").setLevel(logging.DEBUG)
logging.getLogger("examples.pension").setLevel(logging.WARNING)


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


def notest_pensionsim_static():
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

    def reward_from_company(company, individual):
        if company.resources["cash"].number > 0:
            return 1
        else:
            return 0

    # p.PensionInsuranceCompany.determine_client_transaction = make_step(
    #     # [Curr Client Age, Company funds, Reputation, Num Clients]
    #     # (normalized to make broadly usable)
    #     observation_space=spaces.Box(
    #         low=np.array([0.0] * 4),
    #         high=np.array([1.0] * 4)
    #     ),
    #     observation_space_mapping=obs_from_company,
    #     action_space=spaces.Box(low=0.0, high=1.0, shape=(1,)),
    #     action_space_mapping=lambda w: min(max(0, w * 1000), 1000),
    #     reward_mapping=reward_from_company
    # )(p.PensionInsuranceCompany.determine_client_transaction)

    # s = p.PensionSim(max_individuals=10)
    # env_cls = generate_env(s, "examples.pension.PensionInsuranceCompany.determine_client_transaction")
    # register(
    #     id="PensionDemo-v0",
    #     entry_point=env_cls,  # "env_def:env_cls",
    # )

    import gym
    env = gym.make("PensionExample-v0")
    env.reset()
    rew = 0
    obs = [0.0] * 4
    done = False
    while not done:
        action = 0.5  # static for now
        obs, rew, done, info = env.step(action)


def learn(agent, episodes, max_steps):
    overall = 0
    last_100 = np.zeros((100,))
    num_actions = agent.q_function.action_disc.space.n
    for episode in range(episodes):
        logger.debug('size of q table: %s',
                     len(agent.q_function.q_table.keys()) * num_actions)

        q_table, cumul_reward, num_steps, info = \
            agent.run_episode(max_steps=max_steps, exploit=False)

        overall += cumul_reward
        last_100[episode % 100] = cumul_reward
        logger.warning('Episode %s finished after %s timesteps with '
                       'cumulative reward %s (last 100 mean = %s)',
                       episode, num_steps, cumul_reward, last_100.mean())
        if type(agent.env).__name__ == 'PensionEnv':
            logger.warning('year %s, q table size %s, epsilon %s, alpha %s, '
                           '#humans %s, reputation %s',
                           agent.env.year,
                           len(q_table.keys()) * num_actions,
                           agent.epsilon, agent.alpha,
                           len([h for h in agent.env.humans if h.active]),
                           info['company'].reputation)
        else:
            logger.warning(
                'q table size %s, epsilon %s, alpha %s',
                len(q_table.keys()), agent.epsilon, agent.alpha)

    logger.warning('Overall cumulative reward: %s', overall / episodes)
    logger.warning('Average reward last 100 episodes: %s', last_100.mean())
    return q_table


def test_pension_q_learn():
    import gym
    import examples  # registers the env
    from agents import q_agent, value_function

    env = gym.make("PensionExample-v0")

    num_bins = 12
    log_bins = True

    q_func = value_function.QFunction(env,
                                      default_value=0,
                                      discretize_bins=num_bins,
                                      discretize_log=log_bins)

    agent = q_agent.Agent(env,
                          q_function=q_func,
                          update_policy=q_agent.greedy,
                          exploration_policy=q_agent.epsilon_greedy,
                          gamma=0.99,
                          min_alpha=0.1,
                          min_epsilon=0.1,
                          alpha_decay=1,   # default 1 = fixed alpha (min_alpha)
                          epsilon_decay=1  # default: 1 = fixed epsilon (instant decay)
                          )

    q_table = learn(agent, episodes=1000, max_steps=20000)

    logger.info('###### TESTING: ######')

    logger.setLevel(logging.INFO)

    for _ in range(3):
        reward = agent.run_episode(exploit=True)[1]
        logger.info("reward: %s", reward)


if __name__ == "__main__":
    func_name: str
    for func_name in dir(sys.modules[__name__]):
        func = getattr(sys.modules[__name__], func_name)
        if callable(func) and func_name.startswith("test_"):
            print(f"Calling {func_name}:")
            func()
            print("###############################")
