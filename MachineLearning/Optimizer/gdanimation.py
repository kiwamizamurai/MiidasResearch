import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation  # アニメーション作成のためのメソッドをインポート

fig = plt.figure()
anim = []  # アニメーション用に描くパラパラ図のデータを格納するためのリスト


# y = x^4 - x^3
def function(x):
    return x ** 4 - 2 * (x ** 3) + 1


# minimum: y = -11/16, x = 3/2
def deriv(x):
    return 4 * (x ** 3) - 6 * (x ** 2)


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
        if abs(x - x_) < eps:  # convergence condition
            break
        x = x_
        x_history.append(x)

        # 時刻tにおける質点と，時刻tに至るまでの運動の軌跡の二つの絵を作成し， アニメーション用のリストに格納する。
        tracks = np.array(x_history)
        im = plt.plot(tracks, function(tracks), 'x', linestyle='dashed', color='red', markersize=5, linewidth=2, aa=True)
        anim.append(im)
    return np.array(x_history)


etas = [0.05, 0.1, 0.2, 0.3]
'''
We check the behavior of Gradient Descent compared with the different eta
'''

for i, eta in enumerate(etas):
    tracks = GD(deriv, 2.0, eta)

    plt.subplot(2, 2, (i + 1))
    plt.title("eta: " + str(eta) + ", iter: " + str(len(tracks)))

    x = np.arange(-1, 3.0, 0.01)
    y = function(x)

    plt.plot(x, y, linestyle="-", c="black")


anim = ArtistAnimation(fig, anim)  # アニメーション作成

plt.show()
anim.save("/Users/kiwamizamurai/Desktop/MiidasResearch/t.gif", writer='imagemagick')