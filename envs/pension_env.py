# from numpy.random import normal
import numpy as np
from gym import core, spaces
from gym.utils import seeding
from scipy.stats import norm
import random


def softmax(x):  # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class PensionEnv(core.Env):
    """
    The environment.
    TODO: doc, logging, rendering?
    """

    metadata = {
        'render.modes': ['human', 'ascii']  # todo
    }

    YEARS_IN_EPISODE = 750

    def __init__(self):
        self.companies = []
        self.humans = []
        self.currHumanIdx = 0
        self.year = 0
        self.lastYear = -1

        self.viewer = None
        self.logger = None  # sys.stderr
        # observation: [Human's age, Company's funds, reputation, number of clients]
        high = np.array([100, 1000000])#, 0, 50])
        low = np.array([0, -1000000])#, -5000, 0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.action_space = spaces.Box(low=-100000, high=100000, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.companies = [self.InsuranceCompany()]
        self.humans = [self.Client(self)]
        self.currHumanIdx = 0
        self.year = 0
        return self._get_ob()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """

        action = -1500 if action == 0 else 12000

        reward = 0

        currHuman = self.humans[self.currHumanIdx]
        if currHuman.active:
            if action < 0:
                currHuman.products[0].doDebitPremium(-action, currHuman)
            else:
                currHuman.products[0].doPayOut(action, currHuman)

            currHuman.liveOneYear()

        # removeTheDead() # disabled to keep humans from changing list indices

        # Determine current observation for returning
        # WARNING: THIS SHOULD HAPPEN HERE, BUT AS WE'RE VISITING A DIFFERENT
        # CLIENT EACH TIME, THE LAST OBSERVATION IS ABOUT A DIFFERENT CLIENT
        # AS THE CURRENT OBSERVATION. THE FIX IS TO ACTUALLY N-O-T RETURN THE
        # CURRENT OBSERVATION, BUT THE N-E-X-T CLIENT'S OBSERVATION (SEE BELOW).

        # observation = self._get_ob()
        info = {  # For debugging the environment or understanding results only.
                  # Off-limits for the RL algorithm!
            "year": self.year,
            "human": currHuman,
            "company": self.companies[0]
        }

        # Prepare next step (find next living Client)
        yearChanged = self._next_human()
        livingHumans = len([h for h in self.humans if h.active])

        if livingHumans > 0:
            while not self.humans[self.currHumanIdx].active \
                    and not self._terminal():
                yearChanged |= self._next_human()

        if yearChanged or livingHumans == 0:
            self.year += 1
            yearChanged = True

        if yearChanged:
            # only do this once per year:
            self.companies[0].doCompanyThings()

        if self.companies[0].funds < 0:
            reward += 0.0
        else:
            reward += 1.0  # (1.0+1.0) / (livingHumans+1.0)  # reward for staying alive

        if yearChanged:  # Human(s) decide whether to become clients
            # 50% chance for new customer on reputation 0
            getsNewCustomer = random.random() < norm.cdf(self.companies[0].reputation, loc=0, scale=1500)
            if getsNewCustomer:
                # reward += 10
                self.humans.append(self.Client(self))
                if self.logger:
                    print(self.year, "New customer, reputation: ", self.companies[0].reputation, file=self.logger)

            elif self.logger:
                print(self.year, "A human scorned our company, reputation: ", self.companies[0].reputation, file=self.logger)

        # Use NEXT observation for returning
        # WARNING: THIS SHOULD HAPPEN ABOVE, BUT NEEDS TO HAPPEN HERE FOR
        # REASONS STATED EARLIER.
        observation = self._get_ob()

        # for debugging also return the human that this observation is about:
        info["nextHuman"] = self.humans[self.currHumanIdx]
        # print(info["human"].id, info["nextHuman"].id)

        return (observation, reward, self._terminal(), info)

    def render(self, mode='human'):
        pass
        # raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _next_human(self):
        '''Returns True if year is complete, False otherwise'''
        self.currHumanIdx += 1
        if self.currHumanIdx >= len(self.humans):
            self.currHumanIdx = 0
            return True
        return False  # year didn't change

    def _terminal(self):
        return self.year > self.YEARS_IN_EPISODE or self.companies[0].funds < 0

    def _get_ob(self):
        """
        Get observation (state):
        Human's age, Company's funds
        """
        currHuman = self.humans[self.currHumanIdx]
        age = currHuman.age if currHuman.active else 0
        return np.array([age,
                         self.companies[0].funds,
                         self.companies[0].reputation,
                         len([c for c in self.companies[0].clients if c.active])
                         ])

    # Business objects:

    class InsuranceCompany:
        '''The complete company'''
        clientIdSeq = 0
        productIdSeq = 0

        def __init__(self):
            self.funds = 20000
            self.clients = []
            self.reputation = 0

        def addClient(self, client):
            self.clients.append(client)

        def removeClient(self, client):
            self.clients.remove(client)

        def getPensionProduct(self):
            return PensionEnv.PensionProduct(self)

        def doCompanyThings(self):
            if self.reputation < 0:
                self.reputation += 20
            if self.reputation > 0:
                self.reputation = 0
            self.funds -= 2000  # cost to keep doors open

        def damageReputation(self, damage):
            self.reputation += damage


    class Product:
        '''A product'''

        def __init__(self, company):
            self.company = company
            self.id = PensionEnv.InsuranceCompany.productIdSeq
            PensionEnv.InsuranceCompany.productIdSeq += 1

    class PensionProduct(Product):
        '''Generic Pension Product'''

        def __init__(self, company):
            PensionEnv.Product.__init__(self, company)
            self.saved = 0
            self.paidOut = 0

        def doDebitPremium(self, amount, client):
            if (client.funds < amount):
                return 0  # -1  # cannot get premium from client
                if PensionEnv.logger:
                    print(PensionEnv.year, "Client", client.id,
                          "has insufficient funds for premium!",
                          file=PensionEnv.logger)
            else:
                client.funds -= amount
                self.company.funds += amount
                self.saved += amount
                client.lastTransaction = -amount
                return 0  # +self.company.reputation # neutral reward

        def doPayOut(self, amount, client):
            # if (self.company.funds < amount):
            #     return 0 # -100  # +self.company.reputation # cannot pay anything out
            #     if PensionEnv.logger:
            #         print(PensionEnv.year, "Company", self.company.id,
            #               "has insufficient funds!", file=PensionEnv.logger)
            # else:
            self.company.funds -= amount
            client.funds += amount
            self.paidOut += amount
            client.lastTransaction = amount
            return 0  # +self.company.reputation # neutral reward

    class Client():
        '''A client'''

        def __init__(self, envObj):
            self.env = envObj
            self.id = PensionEnv.InsuranceCompany.clientIdSeq
            PensionEnv.InsuranceCompany.clientIdSeq += 1
            self.products = []
            self.age = 20
            self.funds = 20000
            self.income = 20000
            self.livingExpenses = 15000
            self.lastTransaction = 0
            self.expectation = 0.6 * self.income
            self.happiness = 0
            self.active = True
            c = self.findCompany()
            self.orderProduct(c)

        def findCompany(self):
            reputations = [c.reputation for c in self.env.companies]
            repProbs = softmax(reputations)
            company = np.random.choice(self.env.companies, p=repProbs)
            return company

        def orderProduct(self, company):
            if (self not in company.clients):
                company.addClient(self)
                p = company.getPensionProduct()
                self.products.append(p)

        def cancelProduct(self, product):
            if (product in self.products):
                product.company.removeClient(self)
                self.products.remove(product)

        def liveOneYear(self):
            happinessChange = 0
            self.funds += self.income

            if len(self.products) == 0 and self.age < 63:
                c = self.findCompany()
                self.orderProduct(c)

            if self.age == 67:
                self.income = 0

            if self.age > 25 and self.funds < self.income * 1.75:
                happinessChange -= 20
                if self.env.logger:
                    print(self.env.year, "Human", self.id, "got debited too much w.r.t. funds of {}!".format(self.income), self.happiness+happinessChange, file=self.env.logger)

            if self.age > 67 and self.lastTransaction < self.expectation:
                happinessChange -= 20
                if self.env.logger:
                    print(self.env.year, "Human", self.id, "received {}, less than expectation of {}!".format(self.lastTransaction, self.expectation), self.happiness+happinessChange, file=self.env.logger)

            if self.funds <= 0:
                happinessChange -= 100
                if self.env.logger:
                    print(self.env.year, "Human", self.id, "is broke!", self.happiness+happinessChange, file=self.env.logger)

            if happinessChange >= 0 and self.happiness < 0:
                self.happiness /= 2
                if self.happiness >= -5:
                    self.happiness = 0
            else:
                self.happiness += happinessChange

            leaving = False
            reputation = None
            for p in self.products:
                if (self.happiness < 0):  # Note: can only happen if >=67
                    p.company.damageReputation(self.happiness)
                angryEnough = random.random() > norm.cdf(p.company.reputation, loc=-1500, scale=500)
                leaving |= self.age < 67 and angryEnough
                reputation = p.company.reputation
                # if leaving:
                #     print("HUMAN IS LEAVING, happiness", self.happiness)

            self.funds -= self.livingExpenses

            # print("Happiness", self.happiness, self.age)
            died = random.random() < norm.cdf(self.age, loc=85, scale=10)  # bad approx
            self.active = not (leaving or died)
            if not self.active and self.env.logger:
                what = "Goodbye" if leaving else "RIP"
                print(self.env.year, what, "Human", self.id, "aged", self.age, "happiness", self.happiness, "reputation", reputation, file=self.env.logger)

            self.age += 1

    def removeTheDead(self):
        for h in self.humans:
            if not h.active:
                for p in h.products:
                    p.company.removeClient(h)
                    self.humans.remove(h)
                    if self.env.logger:
                        print(self.year, "RIP simulated human #", h, "- humans left:", len(self.humans), file=self.env.logger)
# end PensionEnv
