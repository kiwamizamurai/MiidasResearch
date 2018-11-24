import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/Users/kiwamizamurai/Desktop/gits/MiidasResearch/')
from pathlib import Path
from config import config

fig_dir_path = Path(config.fig_dir_path)


# y = 3x - 4
# m is slope, b is y-intercept

N = 100
x = 3 * np.random.rand(N, 1)
y = 3 * x - 4 * np.ones_like(x) + 0.5 *  np.random.rand(N, 1)

param = np.random.rand(2)

def model(m, b):
    return m * x + b

def error(m, b):
    sume = np.sum((y - model(m, b)) ** 2)
    return sume / N

costs = [error(*param)]

def train(eta=0.03, iter=500):
    eps = 1e-5
    param_history = param
    for i in range(iter):
        param[0] -= eta * (-2/N) * np.sum(x * (y - model(*param)))
        param[1] -= eta * (-2/N) * np.sum((y - model(*param)))
        costs.append(error(*param))
        param_history = np.concatenate([param_history, param], axis=0)
        if i >= 1 and abs(costs[i+1] - costs[i]) < eps:
            break
    return param_history.reshape(-1, 2)

ph = train()

figd = plt.figure(figsize=(12, 8))
ax = figd.add_subplot(111)
ax.scatter(x, y, c="blue")
#for pa in ph:
#    ax.plot(x, model(*pa), linestyle="-", lw=0.5, alpha=0.7)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(range(len(costs)), costs)
ax1.set_title('Cost: {}'.format(costs[-1]))
ax1.set_xlabel('iteration')
ax1.set_ylabel('cost')
ax2.scatter(ph[:, 0], ph[:, 1], s=3)
ax2.set_title('Param: {}'.format(param))
ax2.scatter(3, -4, c='r')
ax2.set_xlabel('m')
ax2.set_ylabel('b')

#plt.show()

name = fig_dir_path / "linearegression.png"
plt.savefig(name)