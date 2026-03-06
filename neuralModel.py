import numpy as np
from layer import Layer
from testDataOpener import training_data, validation_data
from timer import timeit
import random

'''
allow for neural network to allow inputs in the form of batches of vectors instead of just one vector

change loss function to calculate loss for the batch

vectorise ReLU and everything else tbh

'''

class NeuralNetwork:
    def __init__(self):
        self.input_batch = None
        self.first_layer = Layer(784, 128)
        self.second_layer = Layer(128, 64)
        self.output_layer = Layer(64, 10, True)
        
    def input(self, arr):    
        self.input_batch = np.transpose(np.array(arr))

        if len(self.input_batch.shape) == 1:
            self.input_batch = self.input_batch.reshape(784,1)
            
    def forward_propogation(self):
        # self.first_layer.forward(self.input_batch)
        # self.second_layer.forward(self.first_layer.a_neurons)
        # self.output_layer.forward(self.second_layer.a_neurons)

        self.first_layer.forward(self.input_batch)
        assert not np.isnan(self.first_layer.a_neurons).any(), "NaN in layer 1"
        self.second_layer.forward(self.first_layer.a_neurons)
        assert not np.isnan(self.second_layer.a_neurons).any(), "NaN in layer 2"
        self.output_layer.forward(self.second_layer.a_neurons)
        assert not np.isnan(self.output_layer.a_neurons).any(), "NaN in output"



    def output(self):

          return np.argmax(self.output_layer.a_neurons, axis = 0)
    
    def one_hot_encoding(self, indexes):
        one_hot_encoded = np.zeros((indexes.size, 10))
        one_hot_encoded[np.arange(indexes.size), indexes] = 1

        return np.transpose(one_hot_encoded) 
    
    def cross_entropy_cost(self, exp_batch, act_batch):
        exp_batch = self.one_hot_encoding(exp_batch)
        epsilon = 1e-12
        act_batch = np.clip(act_batch, epsilon, (1. - epsilon))
        prob_dist = -np.mean(np.sum(exp_batch * np.log(act_batch), axis = 0, keepdims = True), axis = 0, keepdims = True)
        return prob_dist
    
    def softmax_cross_entropy_grad(self, exp_batch):
        exp_batch = self.one_hot_encoding(exp_batch)
        return np.subtract(self.output_layer.a_neurons,  exp_batch)
     
    def backward_propogation(self, delta, layer = 3):
        curr_layer = [None, self.first_layer, self.second_layer, self.output_layer][layer]
        prev_layer = [None, self.first_layer, self.second_layer, self.output_layer][layer - 1]

        weights_transpose = curr_layer.weights.T
        
        z_l = prev_layer.z_neurons
        relu_grad = (z_l > 0).astype(float)

        return np.dot(weights_transpose, delta) * relu_grad
    
    def weight_grad(self, batch, layer, delta_l, batch_size):
        prev_layer = [self.input_batch, self.first_layer, self.second_layer, self.output_layer][layer - 1]

        if layer == 1:
            weight_gradients = np.dot(delta_l, prev_layer.T) / batch_size
        else:
            weight_gradients = np.dot(delta_l, prev_layer.a_neurons.T) / batch_size

        return weight_gradients
    
    def bias_grad(self, batch, delta_l, batch_size):
        bias_gradients = np.sum(delta_l, axis = 1, keepdims=True) / batch_size
        return bias_gradients

    def save_parameters(self):
        np.savez(
            'params.npz', 
            W1 = self.first_layer.weights, B1 = self.first_layer.biases,
            W2 = self.second_layer.weights, B2 = self.second_layer.biases,
            W3 = self.output_layer.weights, B3 = self.output_layer.biases,
            )
   
    def load_parameters(self):
        with np.load('params.npz') as data:
            self.first_layer.weights = data['W1']
            self.first_layer.biases = data['B1']
            self.second_layer.weights = data['W2']
            self.second_layer.biases = data['B2']
            self.output_layer.weights = data['W3']
            self.output_layer.biases = data['B3']

    @timeit
    def stoch_grad_descent(self, learning_rate, training_data, batch_size, epoch):
        for _ in range(epoch):
            # shuffle training data
            sample_data = training_data[0]
            exp_sample_data = training_data[1]

            training_data_z = list(zip(sample_data, exp_sample_data))
            random.shuffle(training_data_z)

            sample_data_shuffled , exp_sample_data_shuffled = zip(*training_data_z)
            sample_data_shuffled = np.array(sample_data_shuffled)
            exp_sample_data_shuffled = np.array(exp_sample_data_shuffled)

            curr_sample_ptr = 0

            while (curr_sample_ptr + batch_size) < 50000:
                batch = sample_data_shuffled[curr_sample_ptr : curr_sample_ptr + batch_size]
                exp_batch = exp_sample_data_shuffled[curr_sample_ptr : curr_sample_ptr + batch_size]

                self.input(batch)
                self.forward_propogation()

                delta_l = self.softmax_cross_entropy_grad(exp_batch)

                for i, layer in enumerate(["output_layer", "second_layer", "first_layer"]):
                    bias_grad = learning_rate * self.bias_grad(batch, delta_l, batch_size)
                    weight_grad = learning_rate * self.weight_grad(batch, (3-i), delta_l, batch_size)
                    getattr(self, layer).weights -= weight_grad
                    getattr(self, layer).biases -= bias_grad
                    if i != 2:
                        delta_l = self.backward_propogation(delta_l, (3 - i))

                curr_sample_ptr += batch_size


test_script = True

if __name__ == "__main__" and not test_script:
    batch = training_data[0][0:4]
    exp_batch = training_data[1][0:4]

    model = NeuralNetwork() 
    

    model.input(training_data[0][0:100])
    model.forward_propogation()
    
    
    print(np.exp(np.mean(-model.cross_entropy_cost(training_data[1][0:100], model.output_layer.a_neurons))))

    model.stoch_grad_descent(0.01, training_data, 32, 40)

    model.input(training_data[0][0:100])
    model.forward_propogation()
    
    model.save_parameters()

    print(np.exp(np.mean(-model.cross_entropy_cost(training_data[1][0:100], model.output_layer.a_neurons))))
import time

if __name__ == "__main__" and test_script:
    model = NeuralNetwork()

    batch = training_data[0][0:10]
    exp_batch = training_data[1][0:10]

    model.load_parameters()

    tc = 0

    while True:
        model.input(validation_data[0][tc])
        model.forward_propogation()

        print(f"Exp, {validation_data[1][tc]}, Actual, {model.output()}", end = "")
        if validation_data[1][tc] != model.output()[0]:
            print("            FAIL")
        print("\n")

        time.sleep(0.1)

        tc += 1
