import numpy as np
from layer import Layer
from testDataOpener import training_data
import math


# input of a vector in R^748

# hidden layers:


# first hidden layer, 128 neurons
# second hidden layer, 64 neurons

# output of a vector in R^10

def ReLU(z):
    return max(0, z)

def softmax(logits):
    max_logit = max(logits)
    exp_values = [math.exp(z - max_logit) for z in logits]
    total = sum(exp_values)
    probabilities = [v / total for v in exp_values]
    return probabilities

class NeuralNetwork:
    def __init__(self):
        self.input_vector = None
        self.first_layer = Layer(784, 128)
        self.second_layer = Layer(128, 64)
        self.output_layer = Layer(64, 10)
        

    def input(self, arr):
        #initialise numpy array to represent vector in R^784
        self.input_vector = np.array(arr)

    def forward_propogation(self):
        self.first_layer.a_neurons = np.array([ReLU(i)for i in np.dot(self.input_vector, self.first_layer.weights) + self.first_layer.biases])
       
        self.second_layer.a_neurons = np.array([ReLU(i) for i in np.dot(self.first_layer.a_neurons, self.second_layer.weights) + self.second_layer.biases])
        
        self.output_layer.a_neurons = softmax(np.dot(self.second_layer.a_neurons, self.output_layer.weights) + self.output_layer.biases)

    def output(self):
        return np.argmax(self.output_layer.a_neurons)

    def cross_entropy_cost(self, expected_outcome, actual_vector):
        expected_vector = np.array([0 if i != expected_outcome else 1 for i in range(10)])
        epsilon = 1e-12
        actual_vector = np.clip(actual_vector, epsilon, (1. - epsilon))

        #print(expected_vector)
        return -np.mean(np.sum(expected_vector * np.log(actual_vector)))



if __name__ == "__main__":
    net = NeuralNetwork()

    costs = []

    for i in range(50000):
        actual_value = training_data[1][i]
        data = training_data[0][i]

        net.input(data)
        net.forward_propogation()

        outputVector = net.output_layer.a_neurons

        costs.append(net.cross_entropy_cost(actual_value, outputVector))

    print(np.min(np.array(costs)))
    print(np.mean(np.array(costs)))
    print(np.max(np.array(costs)))
        
