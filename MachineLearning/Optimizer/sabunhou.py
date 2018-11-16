import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
sns.set_style("whitegrid")


fig = plt.figure()
ax = fig.add_subplot(111)

#
# fig2 = plt.figure()
# ax1 = fig2.add_subplot(211)
# ax2 = fig2.add_subplot(212)
#
#
#
# sigmoid = lambda x: 1 / (1 + np.exp(-x))
# derivative = lambda x: (1 / (1 + np.exp(-x))) * (1 - ( 1 / (1 + np.exp(-x))))
#
# x = np.linspace(-10, 10, 100)
#
# ax.plot(x, sigmoid(x))
# ax.set_title("Sigmoid")
#
# ax1.plot(x, derivative(x))
# ax1.set_title("Derivative")
#
#
# ax2.plot(x, norm.pdf(x, 0, 1))
# ax2.set_title("StandardGauss")

sigmoid = lambda x: 1 / (1 + np.exp(-x))
derivative = lambda x: (1 / (1 + np.exp(-x))) * (1 - ( 1 / (1 + np.exp(-x))))

def sabun(func, x):
    eps = 1e-5
    return (func(x+eps) - func(x-eps))/(2*eps)

x = np.linspace(-10, 10, 100)
ax.plot(x, sabun(sigmoid, x), linestyle='-', c="r", label="Sabun")
ax.plot(x, derivative(x), linestyle='-', c="b", label="derivative")
ax.legend()
plt.show()
