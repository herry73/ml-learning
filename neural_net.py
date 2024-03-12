import numpy as np

#sigmoid func
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