import numpy as np
import matplotlib.pyplot as plt

# y = x^4 - 2x^3 + 1
def function(x):
    return x**4 - 2 * (x**3) + 1


def deriv(x):
    return 4 * (x**3) - 6 * (x**2)


def GD(deriv, init, eta, iter=100):
    '''
    :param deriv: derivative of the function you want to optimize
    :param init: start point, initial value
    :param eta: leaning rate, step size
    :return: history
    '''
    eps = 1e-5
    x = init
    x_history = [init]
    for i in range(iter):
        x_ = x - eta * deriv(x)
        print(x)
        if abs(x - x_) < eps:   # convergence condition
            break
        x = x_
        x_history.append(x)
    return np.array(x_history)


etas = [ 0.05, 0.1, 0.2, 0.3]
'''
We check the behavior of Gradient Descent compared with the different eta
'''

for i, eta in enumerate(etas):
    tracks = GD(deriv, 2.0, eta)

    plt.subplot(2, 2, (i+1))

    plt.title("eta: " + str(eta) + ", iter: " + str(len(tracks)))

    x = np.arange(-1, 3.0, 0.01)
    y = function(x)
    plt.plot(x, y, linestyle="-", c="black")
    plt.plot(tracks, function(tracks), 'x', c="r")


#plt.tight_layout()
plt.show()