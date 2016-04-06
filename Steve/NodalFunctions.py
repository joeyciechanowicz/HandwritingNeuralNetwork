import numpy


class NodalFunctions:

    @staticmethod
    def logistic(z):
        return 1.0/(1.0 + numpy.exp(-z))
