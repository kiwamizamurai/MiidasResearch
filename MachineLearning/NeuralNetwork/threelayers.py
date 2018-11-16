import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

label = y
for i in range(len(label)):
    if label[i] == 2:
        label[i] = 0



class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        #self.output = np.zeros(y.shape)

    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv == False:
            return 1 / (1 + np.exp(-x))
        if deriv == True:
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backprop(self):
        eps = 0.005
        d_weights2 = np.dot(self.layer1.T, (-2*(self.y - self.output) * self.sigmoid(self.output, True)))
        d_weights1 = np.dot(self.input.T,  (np.dot(-2*(self.y - self.output) * self.sigmoid(self.output, True), self.weights2.T) * self.sigmoid(self.layer1, True)))

        self.weights1 -= eps * d_weights1
        self.weights2 -= eps * d_weights2

    def train(self):
        self.output = self.feedforward()
        self.backprop()

    def predict(self, data):
        self.layer1 = self.sigmoid(np.dot(data, self.weights1))
        pred = self.sigmoid(np.dot(self.layer1, self.weights2))
        return pred

NN = NeuralNetwork(X, y)

loss = []
for i in range(5000):
    NN.train()
    loss.append(round(np.mean(np.square(y - NN.feedforward())), 3))

plt.figure()
plt.plot(range(len(loss)), loss)

print(y[3])
print(NN.predict(X[3]))

print(y[83])
print(NN.predict(X[83]))

plt.show()