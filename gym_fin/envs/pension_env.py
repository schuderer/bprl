from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Stdlib imports
import logging

# Third party imports
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# Application imports
from gym_fin.envs import utils


# logging.debug("some invisible debug message to get logging ""
#               "to set up implicit stuff. whatever.")
# logging.basicConfig()
logger = logging.getLogger(__name__)
# todo: need to change logging so something else to be gym compatible!

YEARS_IN_EPISODE = 750
MEAN_STOCKS_RETURN = 8.5 / 100  # s&p500
STOCKS_VOLATILITY = 16.5 / 100  # volatility
MEAN_BONDS_RETURN = 3.0 / 100  # short-term bonds
BONDS_VOLATILITY = 5.0 / 100

np_random = None


class PensionEnvError(RuntimeError):
    pass


class PensionEnv(gym.Env):
    """The environment.
    TODO: doc, rendering?
    """

    metadata = {"render.modes": ["human", "ascii"]}  # todo

    def __init__(self):
        self._companies = []
        self.humans = []
        self._curr_human_idx = 0
        self.year = -1

        self.viewer = None
        # observation: [Human's age, Company's funds, reputation,
        #               number of clients]
        high = np.array([100, 1000000])  # , 0, 50])
        low = np.array([0, -1000000])  # , -5000, 0])
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        # self.action_space = spaces.Box(low=-100000, high=100000, shape=(1,),
        #                                dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.investing = True
        # self._cached_new_cdf = {}
        self.seed()

    @property
    def companies(self):
        return self._companies

    def seed(self, seed=None):
        global np_random
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the state of the environment, returning
        an initial observation.

        Returns:
            observation: the initial observation of the space.
            (Initial reward is assumed to be 0.)
        """
        self._companies = [InsuranceCompany(investing=self.investing)]
        self.humans = [Client.maybe_make_client(self, force=True)]
        self._curr_human_idx = 0
        self.year = 0
        return self._get_observation()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's
        internal state.

        Args:
            action: an action provided by the environment

        Returns:
            - (observation, reward, done, info)
            - observation: agent's observation of the current environment
            - reward: float, amount of reward due to the previous action
            - done: a boolean, indicating whether the episode has ended
            - info: a dictionary containing other diagnostic information
                    from the previous action
        """

        if self.year < 0:
            raise PensionEnvError("Called step() before reset()")

        action = -1500 if action == 0 else 12000

        reward = 0

        curr_human = self.humans[self._curr_human_idx]
        if curr_human.active:
            if action < 0:
                if not curr_human.pension_fund.do_debit_premium(
                    -action, curr_human
                ):
                    logger.info(
                        "%s Client %s has insufficient funds for premium!",
                        self.year,
                        curr_human.id,
                    )
            else:
                curr_human.pension_fund.do_pay_out(action, curr_human)

            curr_human.live_one_year()

        # remove_the_dead() #disabled to keep humans from changing list indices

        # Determine current observation for returning
        # WARNING: THIS SHOULD HAPPEN HERE, BUT AS WE'RE VISITING A DIFFERENT
        # CLIENT EACH TIME, THE LAST OBSERVATION IS ABOUT A DIFFERENT CLIENT
        # AS THE CURRENT OBSERVATION. THE FIX IS TO ACTUALLY N-O-T RETURN THE
        # CURRENT OBSERVATION, BUT THE N-E-X-T CLIENT'S OBSERVATION (SEE BELOW)

        # observation = self._get_observation()
        info = {  # For debugging the environment or understanding results only
            # Off-limits for the RL algorithm!
            "year": self.year,
            "human": curr_human,
            "company": self._companies[0],
        }

        # Prepare next step (find next living Client)
        year_changed = self._next_human()
        living_humans = len([h for h in self.humans if h.active])

        if living_humans > 0:
            while (
                not self.humans[self._curr_human_idx].active
                and not self._terminal()
            ):
                year_changed |= self._next_human()

        if year_changed or living_humans == 0:
            self.year += 1
            year_changed = True

        if year_changed:
            # only do this once per year:
            for company in self._companies:
                company.run_company()

        if self._companies[0].funds <= 0:
            reward += 0.0
        else:
            reward += 1.0  # reward for staying alive

        if year_changed:  # Human(s) decide whether to become clients
            client = Client.maybe_make_client(self)
            if client:
                # reward += 10
                self.humans.append(client)
                logger.info(
                    "%s New customer, reputation: %s",
                    self.year,
                    self._companies[0].reputation,
                )
            else:
                logger.info(
                    "%s A human scorned our company, reputation: %s",
                    self.year,
                    self._companies[0].reputation,
                )

        # Use NEXT observation for returning
        # WARNING: THIS SHOULD HAPPEN ABOVE, BUT NEEDS TO HAPPEN HERE FOR
        # REASONS STATED EARLIER.
        observation = self._get_observation()

        # for debugging also return the human that this observation is about:
        info["nextHuman"] = self.humans[self._curr_human_idx]
        # print(info['human'].id, info['nextHuman'].id)

        return (observation, reward, self._terminal(), info)

    def render(self, mode="human", close=False):
        pass
        # raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _next_human(self):
        """Returns True if year is complete, False otherwise"""
        self._curr_human_idx += 1
        if self._curr_human_idx >= len(self.humans):
            self._curr_human_idx = 0
            return True
        return False  # year didn't change

    def _terminal(self):
        return self.year > YEARS_IN_EPISODE or self._companies[0].funds < 0

    def _get_observation(self):
        """Get observation (state)

        Returns:
            List of
            [Human's age, Company's funds, reputation, number of clients]
        """
        curr_human = self.humans[self._curr_human_idx]
        age = curr_human.age if curr_human.active else 0
        return np.array(
            [
                age,
                self._companies[0].funds,
                # self._companies[0].reputation,
                # len([c for c in self._companies[0].clients if c.active]),
            ]
        )

    # def remove_the_dead(self):
    #     for h in self.humans:
    #         if not h.active:
    #             for p in h.products:
    #                 p.company.remove_client(h)
    #                 self.humans.remove(h)
    #                 logger.info('%s RIP simulated human # %s '
    #                             '- humans left: %s',
    #                             self.year, h, len(self.humans))


# Business objects:
class InsuranceCompanyError(ValueError):
    pass


class InsuranceCompany:
    """The complete company"""

    running_cost = -2000
    reputation_recovery = +20

    def __init__(self, investing=True):
        self.investing = investing
        self.funds = 20000
        self.stocksAllocation = 0.7
        # self.bondsAllocation = 1 - self.stocksAllocation
        self.clients = []
        self.reputation = 0

    def create_membership(self, client):
        self.clients.append(client)
        return True  # Currently all clients are accepted

    # def remove_client(self, client):
    #     self.clients.remove(client)

    def run_company(self):
        # Spend cost to keep doors open
        self.funds -= InsuranceCompany.running_cost

        if self.investing:
            # Calculate investment returns
            # Geometric Brownian Motion (Gaussian Process), nicely explained:
            # https://newportquant.com/price-simulation-with-geometric-brownian-motion/
            # Assuming monthly (free) re-weighing
            global np_random
            stocksValue = self.funds * self.stocksAllocation
            bondsValue = self.funds - stocksValue
            stocksReturns = stocksValue * np_random.normal(
                loc=MEAN_STOCKS_RETURN, scale=STOCKS_VOLATILITY
            )
            bondsReturns = bondsValue * np_random.normal(
                loc=MEAN_BONDS_RETURN, scale=BONDS_VOLATILITY
            )
            # print('funds',self.funds,'stocks',stocksValue,'stockReturns',stocksReturns,'bonds',bondsValue,'bondsReturns',bondsReturns)
            self.funds += stocksReturns + bondsReturns

        # Correct reputation
        if self.reputation < 0:
            self.reputation += InsuranceCompany.reputation_recovery
        if self.reputation > 0:
            self.reputation = 0

    def damage_reputation(self, damage):
        if damage <= 0:
            self.reputation += damage
        else:
            raise InsuranceCompanyError("Damage must be negative")

    def do_debit_premium(self, amount, client):
        if amount < 0:
            raise InsuranceCompanyError("Amount must be positive")
        if client.give_or_take(-amount):
            self.funds += amount
            return True  # could get premium from client
        else:
            return False  # cannot get premium from client

    def do_pay_out(self, amount, client):
        # if (self.company.funds < amount):
        #     return 0 # -100  # +self.company.reputation # cannot pay anything
        #     if PensionEnv.logger:
        #         print(PensionEnv.year, 'Company', self.company.id,
        #               'has insufficient funds!', file=PensionEnv.logger)
        # else:
        if amount < 0:
            raise InsuranceCompanyError("Amount must be positive")
        if self.funds >= amount:
            self.funds -= amount
            client.give_or_take(amount)
            return True
        else:
            # For now, also carry out transaction if company would go broke
            self.funds -= amount
            client.give_or_take(amount)
            return False


class Seq:
    """Base class to create (incremental) class instance ids"""

    seq = -1

    @classmethod
    def create_id(cls):
        cls.seq += 1
        return cls.seq


class ClientError(ValueError):
    pass


class Client(Seq):
    """A client"""

    income_end_age = 67

    def __init__(self, envObj):
        self.env = envObj
        self.id = self.create_id()
        self.pension_fund = None
        self.age = 20
        self.funds = 20000
        self.income = 20000
        self.living_expenses = 15000
        self.last_transaction = 0
        self.expectation = 0.6 * self.income
        self.happiness = 0
        self.active = True
        # self._cached_leave_cdf = {}
        # self._cached_death_cdf = {}

    @classmethod
    def maybe_make_client(cls, env, force=False):
        global np_random
        client = cls(env)
        comp = client._find_company()
        # 50% chance for new customer on reputation 0
        if force or np_random.uniform() < client._new_cdf(comp.reputation):
            client._become_client_of(comp)
            return client
        else:
            return None

    @staticmethod
    def _new_cdf(rep):
        # lru_cached:
        return utils.cached_cdf(int(rep / 100) * 100, 0, 1500)

        # uncached
        # return norm.cdf(int(rep), loc=0, scale=1500)

        # self-cached
        # int_val = int(rep)
        # if int_val in self._cached_new_cdf:
        #     return self._cached_new_cdf[int_val]
        # else:
        #     self._cached_new_cdf[int_val] = norm.cdf(int_val, loc=0,
        #                                              scale=1500)
        #     return self._cached_new_cdf[int_val]

    @staticmethod
    def _leave_cdf(rep):
        # lru-cached:
        return utils.cached_cdf(int(rep / 20) * 20, -1500, 500)
        # uncached
        # return norm.cdf(int(rep), loc=-1500, scale=500)

        # self-cached
        # int_val = int(rep)
        # if int_val in self._cached_leave_cdf:
        #     return self._cached_leave_cdf[int_val]
        # else:
        #     self._cached_leave_cdf[int_val] = norm.cdf(int_val, loc=-1500,
        #                                                scale=500)
        #     return self._cached_leave_cdf[int_val]

    @staticmethod
    def _death_cdf(age):
        # lru-cached:
        return utils.cached_cdf(int(age / 2) * 2, 85, 10)

        # uncached
        # return norm.cdf(int(age), loc=85, scale=10)

        # self-cached
        # int_val = int(age)
        # if int_val in self._cached_death_cdf:
        #     return self._cached_death_cdf[int_val]
        # else:
        #     self._cached_death_cdf[int_val] = norm.cdf(int_val, loc=85,
        #                                                scale=10)
        #     return self._cached_death_cdf[int_val]

    def _find_company(self):
        global np_random
        reputations = [c.reputation for c in self.env.companies]
        rep_probs = utils.softmax(reputations)
        try:
            company = np_random.choice(self.env.companies, p=rep_probs)
        except ValueError:
            raise ClientError("There are no companies in the environment.")
        return company

    def _become_client_of(self, company):
        if self.pension_fund is not None:
            raise ClientError(
                "Human {} already client of {}".format(self.id, company)
            )

        if company.create_membership(self):
            self.pension_fund = company
        else:
            raise ClientError(
                "Client membership for {} refused by {}".format(
                    self.id, company
                )
            )

    def _leave(self):
        logger.info(
            "Goodbye %s Human %s aged %s happiness %s reputation %s",
            self.env.year,
            self.id,
            self.age,
            self.happiness,
            self.pension_fund.reputation,
        )
        self.pension_fund = None
        self.active = False

    def _die(self):
        logger.info(
            "RIP %s Human %s aged %s happiness %s reputation %s",
            self.env.year,
            self.id,
            self.age,
            self.happiness,
            self.pension_fund.reputation,
        )
        self.active = False

    def give_or_take(self, amount):
        if amount >= 0 or self.funds >= -amount:
            self.funds += amount
            self.last_transaction = amount
            return True
        else:
            return False

    def live_one_year(self):
        global np_random
        happiness_change = 0
        self.funds += self.income

        if self.age >= Client.income_end_age:
            self.income = 0

        if self.age > 25 and self.funds < self.income * 1.75:
            happiness_change -= 20
            logger.info(
                "%s Human %s got debited too much w.r.t. funds of %s! %s",
                self.env.year,
                self.id,
                self.income,
                self.happiness + happiness_change,
            )

        if (
            self.age > Client.income_end_age
            and self.last_transaction < self.expectation
        ):
            happiness_change -= 20
            logger.info(
                "%s Human %s received %s, less than expectation of %s! %s",
                self.env.year,
                self.id,
                self.last_transaction,
                self.expectation,
                self.happiness + happiness_change,
            )

        if self.funds <= 0:
            happiness_change -= 100
            logger.info(
                "%s Human %s is broke! %s",
                self.env.year,
                self.id,
                self.happiness + happiness_change,
            )

        if happiness_change >= 0 and self.happiness < 0:
            self.happiness /= 2
            if self.happiness >= -5:
                self.happiness = 0
        else:
            self.happiness += happiness_change

        leaving = False
        if self.pension_fund is not None:
            company = self.pension_fund
            if self.happiness < 0:  # Note: can only happen if >=67
                company.damage_reputation(self.happiness)
            _angry_threshold = self._leave_cdf(company.reputation)
            angry_enough = np_random.uniform() > _angry_threshold
            leaving |= self.age < Client.income_end_age and angry_enough
            # if leaving:
            #     Adjusting the regulations
            #     contribTotal = (self.age - 20) * 1500
            #     refund = 0.75 * contribTotal
            #     print('HUMAN IS LEAVING, age', self.age, ', happiness',
            #             self.happiness,
            #             ', contribTotal', contribTotal, ', refund', refund,
            #             ', c.funds before', p.company.funds,
            #             ', h.funds before', self.funds)
            #     p.company.funds -= refund
            #     self.funds += refund
            #     print('after: c.funds', p.company.funds, ', h.funds',
            #           self.funds)

        self.funds -= self.living_expenses

        _death_threshold = self._death_cdf(self.age)
        died = np_random.uniform() < _death_threshold  # bad approx

        if died:
            self._die()
        elif leaving:
            self._leave()
        else:
            self.age += 1
