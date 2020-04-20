#!/usr/bin/python3
import numpy as np

def sigmoid(s):
    return 1/(1 + np.exp(-s))

class NeuralNetwork:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.weights1, self.weights2 = self.new_weights()
        self.bias1 = np.random.rand()
        self.bias2 = np.random.rand()

    def loss_func(self, desired_output, actual_output):
        return (desired_output - actual_output) ** 2

    def calc_output(self, input_data, weights1, weights2):
        hidden_neurons = np.zeros(weights2.shape[0])
        for input_neuron_idx in range(len(weights1)):
            input_neuron_value = input_data[input_neuron_idx]
            for hidden_neuron_idx in range(len(weights1[input_neuron_idx])):
                hidden_neuron_value = input_neuron_value * weights1[input_neuron_idx][hidden_neuron_idx]
                hidden_neurons[hidden_neuron_idx] += hidden_neuron_value
        for idx in range(len(hidden_neurons)):
            hidden_neurons[idx] = sigmoid(hidden_neurons[idx])
        output = None
        for idx in range(len(hidden_neurons)):
            if output is None:
                output = hidden_neurons[idx] * weights2[idx]
            else:
                output += hidden_neurons[idx] * weights2[idx]
        return sigmoid(output)

    def calc_loss(self, input_data, desired_output, weights1, weights2):
        total_loss = self.loss_func(desired_output[0],
                self.calc_output(input_data[0], weights1, weights2))
        for idx in range(1, len(input_data)):
            total_loss += self.loss_func(desired_output[idx],
                    self.calc_output(input_data[idx], weights1, weights2))
        return total_loss

    def train(self, epochs=200):
        for epoch in range(epochs):
            print('epoch {}'.format(epoch))
            new_weights1, new_weights2 = self.new_weights()
            old_weights_loss = self.calc_loss(self.input_data,
                                              self.output_data,
                                              self.weights1,
                                              self.weights2)
            new_weights_loss = self.calc_loss(self.input_data,
                                              self.output_data,
                                              new_weights1,
                                              new_weights2)
            if new_weights_loss < old_weights_loss:
                self.weights1 = new_weights1
                self.weights2 = new_weights2
                print('loss: {}'.format(new_weights_loss))

    def new_weights(self):
        weights1 = np.random.uniform(-1, 1, (1, 4))
        weights2 = np.random.uniform(-1, 1, 4)
        return weights1, weights2

    def predict(self, input_data):
        return self.calc_output(input_data, self.weights1, self.weights2)

if __name__ == '__main__':
    input_data, output_data = [], []
    for i in np.arange(1, 1000, 5):
        input_data.append([i])
        output_data.append(i/1000.)
    network = NeuralNetwork(np.array(input_data), np.array(output_data))
    network.train(epochs=100)
    print('got it! {}'.format(network.predict([100])))
    print('got it! {}'.format(network.predict([200])))
    print('got it! {}'.format(network.predict([900])))
