import numpy as np
from testDataOpener import training_data


class Layer:
    def __init__(self, input_size, output_size, output_layer = False):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.a_neurons = np.zeros(output_size)
        self.z_neurons = None
        self.activation_function = lambda z : self.ReLU(z) if not output_layer else self.softmax(z)

    def forward(self, inputs):
        self.z_neurons = np.dot(self.weights, inputs) + self.biases
        self.a_neurons = self.activation_function(self.z_neurons)
        return self.a_neurons

    def ReLU(self, z):
        return np.maximum(0, z)
    
    def softmax(self, logits):
        max_logit = np.max(logits, axis = 0, keepdims = True)
        exp = np.exp(np.subtract(logits, max_logit))
        return exp / np.sum(exp, axis = 0, keepdims = True)
    
if __name__ == "__main__":
    hidden_layer1 = Layer(784, 128)
    hidden_layer2 = Layer(128, 64)
    output_layer = Layer(64, 10, True)

    batch = training_data[0][0]

    hidden_layer1.forward(batch)
    hidden_layer2.forward(hidden_layer1.a_neurons)
    output_layer.forward(hidden_layer2.a_neurons)

    print(np.sum(output_layer.a_neurons))








