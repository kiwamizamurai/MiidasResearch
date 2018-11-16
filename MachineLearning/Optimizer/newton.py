import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys, os
sys.path.append('/Users/kiwamizamurai/Desktop/gits/MiidasResearch/')
from pathlib import Path
from config import config
sns.set_style("whitegrid")


fig_dir_path = Path(config.fig_dir_path)


fig = plt.figure()
ax = fig.add_subplot(111)

def function(x):
    return x**2 - 2

def prime(x):
    return 2*x

def tyler(x, center):
    return function(center) + prime(center)*(x-center)

x = np.linspace(-5, 10, 300)
y = function(x)


a=4
print("初期値", a)
ax.plot(x, tyler(x, a))

while True:
    a2 =  a - (a**2-2)/(2*a)
    ax.plot(x, tyler(x, a2))
    if abs(a2-a)<0.0001:
        print("収束", a2)
        break
    a = a2
    print("只今", a)




ax.plot(x, y)
ax.plot(x, np.zeros_like(x), c="black")
ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)
#plt.show()

aa = fig_dir_path / "newton.png"

plt.savefig(aa)

