import matplotlib; matplotlib.use("TkAgg")
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


def main():
    hp = {'K': 100,      # Number of workers
          'p': 2,     # pareto exponent
          'dim': 2,     # dimension of optimization problem
          }

    quadratic_function = RandomQuadraticFunction(hp['dim'])
    time_steps = 100000
    p_opt = 1/2*(1 + 1/(hp['p']))
    p_max = 1/hp['p'] + 0.00001
    p_min = 1
    stepsize_exponents = [p_max, p_min, p_opt]

    plt.figure()
    r = 0
    for exponent in stepsize_exponents:
        print(r)
        stepsize = np.array([1/(n + 1)**exponent for n in range(time_steps)])
        variable_trajectory = np.zeros(hp['dim']+1)
        obj_gradient_norm_trajectory = np.zeros((time_steps, 1))

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
            obj_gradient_norm_trajectory[n, :] = quadratic_function.evaluate(variable_trajectory)

        N = 1000  # running mean window
        if r == 0:
            plt.plot(running_mean(obj_gradient_norm_trajectory, N), label=r'$q_{max}$')
        elif r == 1:
            plt.plot(running_mean(obj_gradient_norm_trajectory, N), label=r'$q_{min}$')
        else:
            plt.plot(running_mean(obj_gradient_norm_trajectory, N), label=r'$q_{opt}$')
        r += 1

    plt.xlabel(r'$n$')
    plt.ylabel(r'$ || \nabla_x F(x_n) ||$')
    plt.yscale('log')
    plt.legend()
    plt.savefig('evaluation_compare_avg.pdf')


if __name__ == "__main__":
    main()
