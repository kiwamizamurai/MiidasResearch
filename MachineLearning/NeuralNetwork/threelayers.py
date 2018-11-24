import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append('/Users/kiwamizamurai/Desktop/gits/MiidasResearch/')
from pathlib import Path
from config import config

fig_dir_path = Path(config.fig_dir_path)


class NeuralNetwork:
    def __init__(self, x, y, hn):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], hn)
        self.weights2 = np.random.rand(hn, 1)
        self.y = y

    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv == False:
            return 1 / (1 + np.exp(-x))
        if deriv == True:
            return x * (1-x)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def loss(self):
        return np.mean(np.square(self.y - self.feedforward()))

    def backprop(self):
        eps = 0.005
        d_weights2 = np.dot(self.layer1.T, (-2*(self.y - self.output) * self.sigmoid(self.output, True)))
        d_weights1 = np.dot(self.input.T,  (np.dot(-2*(self.y - self.output) * self.sigmoid(self.output, True), self.weights2.T) * self.sigmoid(self.layer1, True)))

        self.weights1 -= eps * d_weights1
        self.weights2 -= eps * d_weights2

    def accuracy(self):
        return np.sum(np.where(self.predict(self.input) == self.y, 1, 0))/len(self.y)

    def train(self):
        self.output = self.feedforward()
        self.backprop()

    def predict(self, data):
        self.layer1 = self.sigmoid(np.dot(data, self.weights1))
        pred = self.sigmoid(np.dot(self.layer1, self.weights2))
        return np.where(pred < 0.5, 0.0, 1.0)



iris = load_iris()
X = iris.data[:100, :]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

NN = NeuralNetwork(X_train, y_train.reshape(-1, 1), 9)

loss = []
for i in range(200):
    NN.train()
    print('iter: {}'.format(i),', Accuracy: ', NN.accuracy())
    loss.append(NN.loss())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(len(loss)), loss)
ax.set_title('objective function')
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
#plt.show()

print('Pred: ', NN.predict(X[4]), ', Label: ', y[4])

name = fig_dir_path / "3layer.png"
plt.savefig(name)