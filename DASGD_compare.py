import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto
from sklearn.datasets import make_spd_matrix
from scipy.integrate import dblquad

import random


seed = 2024
np.random.seed(seed)
random.seed(seed)


class RandomQuadraticFunction:
    def __init__(self, dim):
        self.dim = dim
        self.A = make_spd_matrix(self.dim)
        self.b = 2*np.random.rand(self.dim)-np.zeros(self.dim)
        self.c = 0

    def call(self, x):
        return np.einsum('i,i', np.einsum('ij,j->i', self.A, x), x) + np.einsum('i,i', self.b, x) + self.c

    def sample(self):
        z = np.random.rand(self.dim)
        return [z, self.call(z)]

    # def evaluate(self, x):
    #     samples = 25
    #     error = 0
    #     for s in range(samples):
    #         z = 2 * (np.random.rand(self.dim) - np.zeros(self.dim) / 2)
    #         sample = [z, self.call(z)]
    #
    #         g_weight = 2 * (np.einsum('i,i->', x[:2], sample[0]) + x[2] - sample[1])
    #         error += g_weight*np.concatenate((sample[0], np.array([1])), axis=0)
    #     return np.linalg.norm(error)

    def evaluate(self, x):
        v1 = 2*dblquad(lambda v, w: (x[0] * v + x[1] * w + x[2] - self.call(np.array([v, w]))) * v, 0, 1, 0, 1)[0]
        v2 = 2*dblquad(lambda v, w: (x[0] * v + x[1] * w + x[2] - self.call(np.array([v, w]))) * w, 0, 1, 0, 1)[0]
        v3 = 2*dblquad(lambda v, w: (x[0] * v + x[1] * w + x[2] - self.call(np.array([v, w]))), 0, 1, 0, 1)[0]

        return np.linalg.norm(np.array([v1, v2, v3]))


class Worker:
    def __init__(self, pareto):
        self.time = 0
        self.x = None
        self.sample = None
        self.p = pareto

    def get_job(self, x, sample):
        self.time += pareto.rvs(self.p, size=1)  # Add computing time
        self.x = x
        self.sample = sample

    def get_gradient(self):
        if self.x is not None and self.sample is not None:
            g_weight = 2*(np.einsum('i,i->', self.x[:2], self.sample[0]) + self.x[2] - self.sample[1])
            return g_weight*np.concatenate((self.sample[0], np.array([1])), axis=0)


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def plot_mean_variance(data, kernel_size_mean, kernel_size_variance, label):
    std_nr = 1
    kernel_mean = np.ones(kernel_size_mean) / kernel_size_mean
    kernel_variance = np.ones(kernel_size_variance) / kernel_size_variance
    mean = np.convolve(np.mean(data, axis=1), kernel_mean , mode='valid')
    standard_dev = np.convolve(np.std(data, axis=1), kernel_variance, mode='valid')
    plt.plot(mean, label=label)
    plt.fill_between(range(len(data[:, 0])-kernel_size_mean+1), mean-std_nr*standard_dev, mean+std_nr*standard_dev, alpha=0.2)

def main():
    hp = {'K': 100,      # Number of workers
          'p': 10/7,     # pareto exponent
          'dim': 2,     # dimension of optimization problem
          }

    quadratic_function = RandomQuadraticFunction(hp['dim'])
    time_steps = 1000000
    runs = 10
    p_opt = 1/2*(1 + 1/(hp['p']))
    p_max = 1/hp['p']
    p_unstable = 0.55
    p_min = 1
    stepsize_exponents = [p_unstable, p_max, p_opt, p_min]
    #stepsize_exponents = [0.55, 0.7, 0.85, 1]
    plt.figure()
    r = 0
    for exponent in stepsize_exponents:
        print(r)
        stepsize = np.array([1/(n + 1)**exponent for n in range(time_steps)])

        obj_gradient_norm_trajectory = np.zeros((time_steps, runs))

        for i in range(runs):
            variable_trajectory = np.zeros(hp['dim'] + 1)
            workers = []
            for k in range(hp['K']):
                if k == 0:
                    workers.append(Worker(hp['p']))
                else:
                    workers.append(Worker(hp['p'] + 4*np.random.rand()))  # workers are heterogeneous with different speed
                workers[k].get_job(variable_trajectory, quadratic_function.sample())

            m = 0
            l = 0
            for n in range(time_steps):
                k = np.argmin([worker.time for worker in workers])
                gradient = workers[k].get_gradient()
                if k > hp['K']*1/4:  # 3/4 of the workers calculate for the first component
                    variable_trajectory[:2] = variable_trajectory[:2] - stepsize[m]*gradient[:2]
                    m += 1
                else:
                    variable_trajectory[2] = variable_trajectory[2] - stepsize[l]*gradient[2]
                    l += 1
                workers[k].get_job(variable_trajectory, quadratic_function.sample())
                obj_gradient_norm_trajectory[n, i] = quadratic_function.evaluate(variable_trajectory)

        kernel_size_mean = 100
        kernel_size_var = 100
        if r == 0:
            plot_mean_variance(obj_gradient_norm_trajectory, kernel_size_mean, kernel_size_var, '$q = 0.55$')
        elif r == 1:
            plot_mean_variance(obj_gradient_norm_trajectory, kernel_size_mean, kernel_size_var, '$q = 0.70$')
        elif r == 2:
            plot_mean_variance(obj_gradient_norm_trajectory, kernel_size_mean, kernel_size_var, '$q = 0.85$')
        else:
            plot_mean_variance(obj_gradient_norm_trajectory, kernel_size_mean, kernel_size_var, '$q = 1.00$')
        r += 1

    plt.ylim([1/10000, 10])
    plt.xlabel(r'$n$')
    plt.ylabel(r'$ || \nabla_x F(x_n) ||$')
    plt.yscale('log')
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(timestamp + 'DASGD_compare.pdf')
    plt.show()


if __name__ == "__main__":
    main()
