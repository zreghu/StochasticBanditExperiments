import numpy as np

def confidence_bound(T, n):
    return np.sqrt(2 * np.log(T) / n)
    
def tau(x, alpha):
    return np.power(1 + alpha, x)
    
def ucb2_confidence_bound(T, n, t, epoch, alpha = 0.0001):
    return np.sqrt((1 + alpha) * np.log(np.exp(1) * t / tau(epoch, alpha)) / (2 * tau(epoch, alpha)))

def ucb_tuned_confidence_bound(T, n, t, x, x_squared):
    def V(t):
        return (1/n * x_squared) - np.power(x, 2) + np.sqrt(2 * np.log(t) / n)
    
    return np.sqrt((np.log(T)/ n) * np.minimum(np.ones(len(n)) * 1/4, V(t)))

def moss_confidence_bound(T, n, t, N):
    return np.sqrt(np.maximum(np.log(T / (n * N)), np.zeros(len(n))) / n)