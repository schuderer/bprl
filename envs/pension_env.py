#from numpy.random import normal
import numpy as np
from gym import core, spaces
from gym.utils import seeding
from scipy.stats import norm
import random
import sys

def softmax(x): # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class PensionEnv(core.Env):
    """
    The environment.
    TODO: doc, logging, rendering?
    """
    
    metadata = {
        'render.modes': ['human', 'ascii'] # todo
    }
    
    YEARS_IN_EPISODE = 300
    
    def __init__(self):
        self.companies = []
        self.humans = []
        self.currHumanIdx = 0
        self.year = 0

        self.viewer = None
        self.logger = None # sys.stderr
        # observation: [Human's age, Human's funds, Company's funds]
        high = np.array([150, 1000000, 1000000])
        low = -high
        low[0] = 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-100000, high=100000, shape=(1,), dtype=np.float32)
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
        currHuman = self.humans[self.currHumanIdx]
        currHuman.liveOneYear()
        
        reward = 0
        if action < 0:
            reward = currHuman.products[0].doDebitPremium(-action, currHuman)
        else:
            reward = currHuman.products[0].doPayOut(action, currHuman)
        
        #removeTheDead() # disabled to keep humans from changing list indices
        self.companies[0].doCompanyThings()
        
        
        
        # Determine current observation for returning
        # WARNING: THIS SHOULD HAPPEN HERE, BUT AS WE'RE VISITING A DIFFERENT
        # CLIENT EACH TIME, THE LAST OBSERVATION IS ABOUT A DIFFERENT CLIENT
        # AS THE CURRENT OBSERVATION. THE FIX IS TO ACTUALLY N-O-T RETURN THE
        # CURRENT OBSERVATION, BUT THE N-E-X-T CLIENT'S OBSERVATION (SEE BELOW).
        #observation = self._get_ob()
        info = { # For debugging the environment or understanding results only.
                 # Off-limits for the RL algorithm!
            "year": self.year,
            "human": currHuman,
            "company": self.companies[0]
        }
        
        # Prepare next step (find next living Client)
        yearChanged = self._next_human_year()
        while not self.humans[self.currHumanIdx].alive and not self._terminal():
            yearChanged |= self._next_human_year()
        
        if yearChanged: # Human(s) decide whether to become clients
            getsNewCustomer = random.random() < norm.cdf(self.companies[0].reputation, loc=0, scale=500)
            if getsNewCustomer:
                self.humans.append(self.Client(self))
        
        # Use NEXT observation for returning
        # WARNING: THIS SHOULD HAPPEN ABOVE, BUT NEEDS TO HAPPEN HERE FOR
        # REASONS STATED EARLIER.
        observation = self._get_ob()
        
        return (observation, reward, self._terminal(), info)
    
    def render(self, mode='human'):
        raise NotImplementedError
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def _next_human_year(self):
        '''Returns True if year changed, False otherwise'''
        self.currHumanIdx += 1
        if self.currHumanIdx >= len(self.humans):
            self.currHumanIdx = 0
            self.year += 1 # dealt with all Clients, year is done
            #print("now in year",self.year)
            return True # year changed
        return False # year didn't change
    
    def _terminal(self):
        return self.year > self.YEARS_IN_EPISODE
    
    def _get_ob(self):
        """
        Get observation (state):
        Human's age, Human's funds, Company's funds
        """
        currHuman = self.humans[self.currHumanIdx]
        return np.array([currHuman.age, currHuman.funds, self.companies[0].funds])
    
    
    # Business objects:
    
    class InsuranceCompany:
        '''The complete company'''
        clientIdSeq = 0
        productIdSeq = 0
        
        def __init__(self):
            self.funds = 0
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
                self.reputation += 5
        
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
                return -1 # cannot get premium from client
                if PensionEnv.logger:
                    print("Client",client.id,"has insufficient funds for premium!", file=PensionEnv.logger)
            else:
                client.funds -= amount
                self.company.funds += amount
                self.saved += amount
                client.lastTransaction = -amount
                return 0+self.company.reputation # neutral reward
    
        def doPayOut(self, amount, client):
            if (self.company.funds < amount):
                return -100+self.company.reputation # cannot pay anything out
                if PensionEnv.logger:
                    print("Company",company.id,"has insufficient funds!", file=PensionEnv.logger)
            else:
                self.company.funds -= amount
                client.funds += amount
                self.paidOut += amount
                client.lastTransaction = amount
                return 0+self.company.reputation # neutral reward
    
    class Client():
        '''A client'''
        def __init__(self, envObj):
            self.env = envObj
            self.id = PensionEnv.InsuranceCompany.clientIdSeq
            PensionEnv.InsuranceCompany.clientIdSeq += 1
            self.products = []
            self.age = 20
            self.funds = 0
            self.income = 20000
            self.lastTransaction = 0
            self.expectation = 0.6 * self.income
            self.mode = 'work'
            self.happiness = 0
            self.alive = True
        
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
                    print("Human",self.id,"got debited too much!", self.happiness+happinessChange, file=self.env.logger)
            
            if self.age > 67 and self.lastTransaction < self.expectation:
                happinessChange -= 20
                if self.env.logger:
                    print("Human",self.id,"received less than expected!", self.happiness+happinessChange, file=self.env.logger)
            
            if self.funds <= 0:
                happinessChange -= 100
                if self.env.logger:
                    print("Human",self.id,"is broke!", self.happiness+happinessChange, file=self.env.logger)
            
            if happinessChange >= 0 and self.happiness < 0:
                self.happiness /= 2
                if self.happiness >= -5:
                    self.happiness = 0
            else:
                self.happiness += happinessChange
            
            for p in self.products:
                if (self.happiness < 0):
                    p.company.damageReputation(self.happiness)
            
            self.funds -= 15000 # living expenses
            
            self.alive = not random.random() < norm.cdf(self.age, loc=85, scale=10) # bad approx
            
            self.age += 1
    
    def removeTheDead():
        for h in humans:
            if not h.alive:
                for p in h.products:
                    p.company.removeClient(h)
                    humans.remove(h)
                    if self.env.logger:
                        print("RIP simulated human #", i, "- humans left:", len(humans), file=self.env.logger)
# end PensionEnv
