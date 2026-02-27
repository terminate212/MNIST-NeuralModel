import numpy as np
from layer import Layer
from testDataOpener import training_data, validation_data

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
        self.first_layer.forward(self.input_batch)
        self.second_layer.forward(self.first_layer.a_neurons)

        self.output_layer.forward(self.second_layer.a_neurons)

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

        #weights_transpose = np.transpose(curr_layer.weights).copy()
        weights_transpose = curr_layer.weights.T
        
        z_l = prev_layer.z_neurons
        relu_grad = (z_l > 0).astype(float)

        return np.dot(weights_transpose, delta) * relu_grad
    
    def weight_grad(self, batch, exp_batch, layer):
        self.input(batch)
        self.forward_propogation()

        delta_3 = self.softmax_cross_entropy_grad(exp_batch)
        delta_2 = self.backward_propogation(delta_3, 3)
        delta_1 = self.backward_propogation(delta_2, 2)

        batch_size = len(batch)

        delta_l = [
            delta_1,
            delta_2,
            delta_3
        ]

        prev_layer = [self.input_batch, self.first_layer, self.second_layer, self.output_layer][layer - 1]

        if layer == 1:
            weight_gradients = np.dot(delta_l[layer - 1], prev_layer.T) / batch_size
        else:
            weight_gradients = np.dot(delta_l[layer - 1], prev_layer.a_neurons.T) / batch_size

        return weight_gradients
    
    def bias_grad(self, batch, exp_batch, layer):
        self.input(batch)
        self.forward_propogation()

        delta_3 = self.softmax_cross_entropy_grad(exp_batch)
        delta_2 = self.backward_propogation(delta_3, 3)
        delta_1 = self.backward_propogation(delta_2, 2)

        batch_size = len(batch)

        delta_l = [
            delta_1,
            delta_2,
            delta_3
        ]


        bias_gradients = np.sum(delta_l[layer - 1], axis = 1, keepdims=True) / batch_size

        return bias_gradients

def trained_model():
    model = NeuralNetwork() 

    learning_rate = 0.1

    layers = ["first_layer", "second_layer", "output_layer"]

    for n in range(5000):

        batch = training_data[0][n*10 : (n+1)*10]
        exp_batch = training_data[1][n*10 : (n+1)*10]

        for layer in range(1, 3):
            layer_string = layers[layer-1]

            bias_grad = learning_rate * model.bias_grad(batch, exp_batch, layer)
            weight_grad = learning_rate * model.weight_grad(batch, exp_batch, layer)

            getattr(model, layer_string).weights -= weight_grad
            getattr(model, layer_string).biases -= bias_grad
    return model



test_script = False

if __name__ == "__main__" and not test_script:
    batch = training_data[0][0:4]
    exp_batch = training_data[1][0:4]

    model = NeuralNetwork() 

    learning_rate = 0.1

    layers = ["first_layer", "second_layer", "output_layer"]

    for n in range(5000):

        batch = training_data[0][n*10 : (n+1)*10]
        exp_batch = training_data[1][n*10 : (n+1)*10]

        for layer in range(1, 3):
            layer_string = layers[layer-1]

            bias_grad = learning_rate * model.bias_grad(batch, exp_batch, layer)
            weight_grad = learning_rate * model.weight_grad(batch, exp_batch, layer)

            getattr(model, layer_string).weights -= weight_grad
            getattr(model, layer_string).biases -= bias_grad

    model.input(training_data[0][0:100])
    model.forward_propogation()
    
    print(np.exp(np.mean(-model.cross_entropy_cost(training_data[1][0:100], model.output_layer.a_neurons))))

if __name__ == "__main__" and test_script:
    model = NeuralNetwork()

    batch = training_data[0][0:10]
    exp_batch = training_data[1][0:10]

    model.input(batch)

    model.forward_propogation()

    weight_gradL3 = model.weight_grad(batch, exp_batch, 3)
    weight_gradL2 = model.weight_grad(batch, exp_batch, 2)
    weight_gradL1 = model.weight_grad(batch, exp_batch, 1)

    print(np.max(weight_gradL3))
    print(np.max(weight_gradL2))
    print(np.max(weight_gradL1))


    print("")

    bias_gradL3 = model.bias_grad(batch, exp_batch, 3)
    bias_gradL2 = model.bias_grad(batch, exp_batch, 2)
    bias_gradL1 = model.bias_grad(batch, exp_batch, 1)

    print(np.max(bias_gradL3))
    print(np.max(bias_gradL2))
    print(np.max(bias_gradL1))