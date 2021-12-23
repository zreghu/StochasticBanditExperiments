import numpy as np
from torch.utils.tensorboard import SummaryWriter
import scipy.integrate as integrate
from scipy.stats import norm
from helper import *

class StochasticBanditAlgorithm:
    """
    Algorithm which plays the Stochastic Multi-Armed Bandit problem.
    """

    def __init__(self, N: int, T: int, name: str, log: SummaryWriter):
        # Properties of Algorithm
        self.N = N 
        self.T = T 
        self.log = log
        self.name = name

        # State of Algorithm
        self.t = 0
        self.mu = np.zeros(N)
        self.n = np.zeros(N)

    """ Make choice of action for given round """
    def make_choice(self) -> int:
        pass

    """ Get result of round so that internal states can get updated """
    def get_reward(self, action: int, reward: int) -> None:
        pass

class ExploringFirst(StochasticBanditAlgorithm):
    """ 
    Exploring First Algorithm
     - explores each action for n times then selects best action based
       on average reward
    """

    def __init__(self, N: int, T: int, name: str, log: SummaryWriter, n: int = 10):
        super().__init__(N, T, name, log)
        self.explore_thresh = N * n 
        if (T < self.explore_thresh):
            raise ValueError('T must be greater than n * N')
        self.n_thresh = n

    def make_choice(self) -> int:
        if (self.t < self.explore_thresh):
            return self.t % self.N
        return np.argmax(self.mu/self.n_thresh)

    def get_reward(self, action: int, reward: int) -> None:
        self.mu[action] += reward
        self.t += 1
        return

class EpsilonGreedy(StochasticBanditAlgorithm):
    """ 
    Epsilon-Greedy Algorithm 
    - choose best current algorithm with probability 1 - epsilon
    - otherwise choose random action 
    """
    def __init__(self, N, T, name, log, epsilon = 0.1):
        super().__init__(N, T, name, log)
        self.epsilon = epsilon

    def make_choice(self) -> int:
        if (np.random.random() < self.epsilon):
            return np.random.randint(self.N)
        return np.argmax(self.mu / self.n)

    def get_reward(self, action: int, reward: int) -> None:
        self.mu[action] += reward
        self.n[action] += 1
        self.t += 1
        return

class SuccessiveElimination(StochasticBanditAlgorithm):
    """ 
    Successive Elimination Algorithm 
    - eliminate actions that have upper confidence bounds 
      that are lower than the lower confidence bound of another action
    """
    def __init__(self, N, T, name, log):
        super().__init__(N, T, name, log)
        self.ucb = np.zeros(N)
        self.lcb = np.zeros(N)
        self.active = np.ones(N, dtype=bool)
   
    def make_choice(self) -> int:
        if self.t < self.N:
            return self.t
        return np.argmax((self.mu / self.n)[self.active])
    
    def get_reward(self, action, reward):
        self.mu[action] += reward
        self.n[action] += 1
        self.t += 1

        self.ucb = self.mu/self.n + confidence_bound(self.T, self.n)
        self.lcb = self.mu/self.n - confidence_bound(self.T, self.n)

        for i in np.arange(self.N)[self.active]:
            if np.any(self.ucb[i] < self.lcb[self.active]):
                self.active[i] = False
        return

class UCB1(StochasticBanditAlgorithm):
    """ 
    UCB1 algorithm, just pick largest confidence bound 
    """ 
    def __init__(self, N, T, name, log):
        super().__init__(N, T, name, log)
        self.ucb = np.zeros(N)

    def make_choice(self) -> int:
        if self.t < self.N:
            return self.t
        return np.argmax(self.ucb)

    def get_reward(self, action, reward):
        self.mu[action] += reward
        self.n[action] += 1
        self.t += 1

        self.ucb = self.mu/self.n + confidence_bound(self.T, self.n)
        return
    
class UCB2(StochasticBanditAlgorithm):
    """ 
    UCB2 algorithm, pick largest confidence bound but slightly modified with epochs
    """

    def __init__(self, N, T, name, log, alpha):
        super().__init__(N, T, name, log)
        self.ucb = np.zeros(N)
        self.epoch = np.zeros(N)
        self.alpha = alpha
        self.epoch_timer = 0
        self.epoch_choice = 0

    def make_choice(self) -> int:
        if self.t < self.N:
            return self.t
        elif self.epoch_timer > 0:
            self.epoch_timer -= 1
            return self.epoch_choice

        return np.argmax(self.ucb)

    def get_reward(self, action, reward):
        self.mu[action] += reward
        self.n[action] += 1
        self.t += 1
        self.epoch[action] += 1

        self.ucb = self.mu/self.n + ucb2_confidence_bound(self.T, self.n, self.epoch, self.alpha)
        if (self.epoch_timer <= 0):
            self.epoch[self.epoch_choice] += 1
            self.epoch_choice = np.argmax(self.ucb)
            self.epoch_timer = int(tau(self.epoch[self.epoch_choice] + 1,self.alpha) - tau(self.epoch[self.epoch_choice],self.alpha))
        return

class UCBTuned(StochasticBanditAlgorithm):
    """ 
    Tuned UCB1 algorithm 
    """
    def __init__(self, N, T, name, log):
        super().__init__(N, T, name, log)
        self.ucb = np.zeros(N)
        self.x_squared = np.zeros(N)

    def make_choice(self) -> int:
        if self.t < self.N:
            return self.t
        return np.argmax(self.ucb)

    def get_reward(self, action, reward):
        self.mu[action] += reward
        self.x_squared[action] += reward**2
        self.n[action] += 1
        self.t += 1

        self.ucb = self.mu/self.n + ucb_tuned_confidence_bound(self.T, self.n, self.t, self.mu/self.n, self.x_squared)
        return

class MOSS(StochasticBanditAlgorithm):
    """ 
    MOSS algorithm
    - similar to UCB1 but with different confidence bound 
    """ 
    def __init__(self, N, T, name, log):
        super().__init__(N, T, name, log)
        self.ucb = np.zeros(N)

    def make_choice(self) -> int:
        if self.t < self.N:
            return self.t
        return np.argmax(self.ucb)

    def get_reward(self, action, reward):
        self.mu[action] += reward
        self.n[action] += 1
        self.t += 1

        self.ucb = self.mu/self.n + moss_confidence_bound(self.T, self.n, self.t, self.N)
        return

class POKER(StochasticBanditAlgorithm):
    """ 
    POKER algorithm 
    """ 
    def __init__(self, N, T, name, log):
        super().__init__(N, T, name, log)
        self.r_1 = np.zeros(N)
        self.r_2 = np.zeros(N)
        self.sigma = np.zeros(N)
        self.p_max = -np.inf
        self.i_max = None
    
    def make_choice(self) -> int:
        if self.t < 2:
            return np.random.choice(np.arange(self.N)[self.n == 0])
        if self.i_max is None:
            return np.argmax(self.mu / self.n)
        return self.i_max
    
    def get_reward(self, action, reward):
        self.mu[action] += reward
        self.n[action] += 1
        self.t += 1
        if self.t < 2:
            return 

        self.r_1[action] += reward 
        self.r_2[action] += reward**2

        self.sigma = np.sqrt((self.r_2 / self.n) - (self.r_1 / self.n)**2)
        q = np.size(self.r_1 > 0)
        i_0 = np.argmax(self.mu / self.n)
        i_1 = i_0
        for j in np.flip(np.argsort(self.mu / self.n)):
            if (np.count_nonzero(self.mu/self.n > (self.mu/self.n)[j]) >= np.sqrt(q)):
                i_1 = j
                break
        delta_mu = ((self.mu/self.n)[i_0] - (self.mu/self.n)[i_1]) / np.sqrt(q)
        mu_star = np.argmax(self.mu/self.n)
        self.p_max = -np.inf 
        self.i_max = None

        for i in range(self.N):
            if self.n[i] > 0:
                mu = (self.mu / self.n)[i]
            else:
                mu = (self.mu/self.n)[self.n > 0].mean()
            
            if self.n[i] > 1:
                sigma = self.sigma[i]
            else:
                sigma = self.sigma[self.n > 1].mean()
            
            p = mu + delta_mu * (self.T - self.t) * integrate.quad(lambda x: norm.pdf(x, mu, sigma / np.sqrt(self.n[i] + 1)),mu_star + delta_mu, np.inf)[0]

            
            if (p > self.p_max):
                self.p_max = p 
                self.i_max = i 
        return

        