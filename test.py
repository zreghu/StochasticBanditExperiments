from torch.utils.tensorboard import SummaryWriter
import numpy as np
from algorithms import *
import matplotlib.pyplot as plt

class StochasticBanditTest:
    def __init__(self, N, T, algorithms, rewards, best, log_name):
        self.algorithms = algorithms
        self.N = N
        self.T = T 
        self.rewards = rewards
        self.log_name = log_name
        for algorithm in self.algorithms:
            algorithm.log = SummaryWriter("{}/{}".format(log_name, algorithm.name))
        
        self.algorithm_dict = []
        for i in range(len(algorithms)):
            self.algorithm_dict.append({"algorithm": algorithms[i], "rewards": 0, "best": 0, "runs": 0, "regret_list": [], "percentage_best_list": []})

        self.best = best
        self.best_reward = 0

    def run(self):
        """ 
        Run the stochastic bandit experiment 
        """
        for i in range(self.T):
            rewards = np.zeros(self.N)
            for j, reward in enumerate(self.rewards):
                rewards[j] = reward()
            self.best_reward += rewards[self.best]
            self.run_trial(i, rewards) 

        for alg_dict in self.algorithm_dict:
            print("Algorithm: {}, with regret {}".format(alg_dict["algorithm"].name, alg_dict["regret_list"][-1]))
            rgb = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(alg_dict["regret_list"], label=alg_dict["algorithm"].name)    

        plt.title("Regret Comparison between MAB algorithms")
        plt.ylabel("Regret")
        plt.xlabel("Trials")
        plt.xscale("log")
        plt.legend()
        plt.show()
    
    def run_trial(self, trial, rewards):
        """ 
        Run a single trial
        """
        for alg_dict in self.algorithm_dict:
            algorithm = alg_dict["algorithm"]
            a, r = self.run_algorithm(algorithm, rewards)
            alg_dict["rewards"] += r
            alg_dict["best"] += (a == self.best)
            alg_dict["runs"] += 1
            algorithm.log.add_scalar("Regret", self.best_reward - alg_dict["rewards"], trial)
            algorithm.log.add_scalar("Percentage_Best", (alg_dict["best"] / (trial + 1)), trial)
            alg_dict["regret_list"].append(self.best_reward - alg_dict["rewards"])
            alg_dict["percentage_best_list"].append(alg_dict["best"] / alg_dict["runs"])

    
    def run_algorithm(self, algorithm, reward):
        """
        Run an algorithm and return the reward
        Return: action, reward
        """
        action = algorithm.make_choice()
        r = reward[action]
        algorithm.get_reward(action, r)
        return action, r

def create_algorithms(N, T):
    """
    Create the algorithms to be tested
    """
    algorithms = []
    log = None
    algorithms.append(ExploringFirst(N, T, "ExploringFirst-10", log, n=10))
    algorithms.append(ExploringFirst(N, T, "ExploringFirst-20", log, n=20))
    algorithms.append(EpsilonGreedy(N, T, "EpsilonGreedy-0.5", log, epsilon=0.5))
    algorithms.append(EpsilonGreedy(N, T, "EpsilonGreedy-0.1", log, epsilon=0.1))
    algorithms.append(EpsilonGreedy(N, T, "EpsilonGreedy-0.2", log, epsilon=0.2))
    algorithms.append(SuccessiveElimination(N, T, "SuccessiveElimination", log))
    algorithms.append(UCB1(N, T, "UCB1", log))
    algorithms.append(UCB2(N, T, "UCB2", log, alpha=0.001))
    algorithms.append(UCBTuned(N, T, "UCBTuned", log))
    algorithms.append(MOSS(N, T, "MOSS", log))
    #algorithms.append(POKER(N, T, "POKER", log))
    return algorithms

def create_distribution(N, distribution_randomizer, distribution_type):
    """
    Create the distribution of rewards
    """
    distribution = []
    best_r = []
    for i in range(N):
        r = distribution_randomizer()
        best_r.append(r)
        distribution.append(lambda: distribution_type(r))
    return distribution, np.argmax(best_r)

def main():
    """
    Run the experiment
    """
    N = 10
    T = 10000
    experiment_name = "trial_5"
    rewards, best = create_distribution(N, lambda: np.random.random(), lambda x: np.random.binomial(1,x))

    rewards = [
        lambda: np.random.normal(0.5),
        lambda: np.random.normal(0.5),
        lambda: np.random.normal(0.5),
        lambda: np.random.normal(0.9),
        lambda: np.random.normal(0.3),
        lambda: np.random.normal(0.2),
        lambda: np.random.normal(0.1),
        lambda: np.random.normal(0.6),
        lambda: np.random.normal(0.5),
        lambda: np.random.normal(0.5),
    ]
    best = 3

    algorithms = create_algorithms(N, T)
    test = StochasticBanditTest(N, T, algorithms, rewards, best, experiment_name)
    test.run()

main()
