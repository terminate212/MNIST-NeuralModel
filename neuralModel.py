import numpy as np
from layer import Layer
from testDataOpener import training_data, validation_data, test_data
from timer import timeit
from tqdm import tqdm
import random

class NeuralNetwork:
    def __init__(self, h_layer1_size, h_layer2_size):
        self.input_batch = None
        self.first_layer = Layer(784, h_layer1_size)
        self.second_layer = Layer(h_layer1_size, h_layer2_size)
        self.output_layer = Layer(h_layer2_size, 10, True)     
        
    def input(self, arr):    
        self.input_batch = np.transpose(np.array(arr))

        if len(self.input_batch.shape) == 1:
            self.input_batch = self.input_batch.reshape(784,1)
            
    def forward_pass(self):
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
     
    def backward_propagation(self, delta, layer = 3):
        curr_layer = [None, self.first_layer, self.second_layer, self.output_layer][layer]
        prev_layer = [None, self.first_layer, self.second_layer, self.output_layer][layer - 1]

        weights_transpose = curr_layer.weights.T
        
        z_l = prev_layer.z_neurons
        relu_grad = (z_l > 0).astype(float)

        return np.dot(weights_transpose, delta) * relu_grad
    
    def weight_grad(self, layer, delta_l, batch_size):
        prev_layer = [self.input_batch, self.first_layer, self.second_layer, self.output_layer][layer - 1]

        if layer == 1:
            weight_gradients = np.dot(delta_l, prev_layer.T) / batch_size
        else:
            weight_gradients = np.dot(delta_l, prev_layer.a_neurons.T) / batch_size

        return weight_gradients
    
    def bias_grad(self, delta_l, batch_size):
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
        for _ in tqdm(range(epoch)):
            sample_data = training_data[0]
            exp_sample_data = training_data[1]

            training_data_z = list(zip(sample_data, exp_sample_data))
            random.shuffle(training_data_z)

            sample_data_shuffled , exp_sample_data_shuffled = zip(*training_data_z)
            sample_data_shuffled = np.array(sample_data_shuffled)
            exp_sample_data_shuffled = np.array(exp_sample_data_shuffled)

            curr_sample_ptr = 0

            while (curr_sample_ptr + batch_size) < len(sample_data_shuffled):
                batch = sample_data_shuffled[curr_sample_ptr : curr_sample_ptr + batch_size]
                exp_batch = exp_sample_data_shuffled[curr_sample_ptr : curr_sample_ptr + batch_size]

                self.input(batch)
                self.forward_pass()

                delta_l = self.softmax_cross_entropy_grad(exp_batch)

                for i, layer in enumerate(["output_layer", "second_layer", "first_layer"]):
                    bias_grad = learning_rate * self.bias_grad(delta_l, batch_size)
                    weight_grad = learning_rate * self.weight_grad((3-i), delta_l, batch_size)
                    getattr(self, layer).weights -= weight_grad
                    getattr(self, layer).biases -= bias_grad
                    if i != 2:
                        delta_l = self.backward_propagation(delta_l, (3 - i))

                curr_sample_ptr += batch_size

    def classification_acc(self, sample_set):
        tc = 0
        fails = 0

        while tc != len(sample_set[0]):
            self.input(sample_set[0][tc])
            self.forward_pass()
            if sample_set[1][tc] != self.output()[0]:
                fails += 1
            tc += 1

        return f"{(1 - (fails / 10000))*100}% Accuracy"


test_script = False

if __name__ == "__main__" and not test_script:

    # HyperParameters
    LEARNING_RATE = 0.001
    EPOCH = 40
    BATCH_SIZE = 32

    HIDDEN_LAYER_1_SIZE = 128
    HIDDEN_LAYER_2_SIZE = 64

    model = NeuralNetwork(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE) 

    print(model.classification_acc(validation_data))

    model.stoch_grad_descent(LEARNING_RATE, training_data, BATCH_SIZE, EPOCH)
    model.save_parameters()

    print(model.classification_acc(validation_data))
