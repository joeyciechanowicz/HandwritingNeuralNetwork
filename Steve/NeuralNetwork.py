import numpy
import json

from Steve.Layers import Layers


class NeuralNetwork:
    def __init__(self, layers: Layers, nodal_function):
        self.nodal_function = nodal_function
        self.layer_sizes = layers.all_layers
        self.biases = [numpy.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def forward(self, input_vector):
        x = [[y] for y in input_vector]
        for b, w in zip(self.biases, self.weights):
            dot_product = numpy.dot(w, x)
            temp = self.nodal_function(dot_product + b)
            x = temp
        return x


