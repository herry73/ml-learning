import numpy as np

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedfwd(self,input):
        total = np.dot(self.weights, input) + self.bias
        return sigmoid(total)
    
weights = np.array([0,1])
bias = np.random.randint(5)
print("Bias =",bias)
n = neuron(weights,bias)

x = np.array([2,3])
print(n.feedfwd(x))

class NeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = neuron(weights ,bias)
        self.h2 = neuron(weights, bias)
        self.output = neuron(weights, bias)

    def feedfwd(self,x):
        out_h1 = self.h1.feedfwd(x)
        out_h2 = self.h2.feedfwd(x)

        out_output = self.output.feedfwd(np.array([out_h1,out_h2]))

        return out_output

network = NeuralNetwork()
x = np.array([2,3])
print(network.feedfwd(x))