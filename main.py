import MNISTParser
import numpy
from Steve.NeuralNetwork import NeuralNetwork
from Steve.NodalFunctions import NodalFunctions
from Steve.Layers import Layers

data = MNISTParser.MNISTParser('./MNISTDataset')
data.load_training(100)
# data.load_testing()

print(data.train_labels[1:60])

layers = Layers(728, [30], 10)
network = NeuralNetwork(layers, NodalFunctions.logistic)

#output = network.forward(numpy.random.choice([0, 1], size=(728,)))
#print(output)

network.mini_batch(1, 10, data.train_images, data.train_labels)
