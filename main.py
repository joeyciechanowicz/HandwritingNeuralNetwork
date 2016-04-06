import MNISTParser
import numpy
from Steve.NeuralNetwork import NeuralNetwork
from Steve.NodalFunctions import NodalFunctions
from Steve.Layers import Layers

data = MNISTParser.MNISTParser('./MNISTDataset')
data.load_training(1)
# data.load_testing()

layers = Layers(728, [30], 10)

network = NeuralNetwork(layers, NodalFunctions.logistic)
output = network.forward(numpy.random.choice([0, 1], size=(728,)))

##load data

print(output)
