#
# Taken from https://github.com/sorki/python-mnist
# Added ability to specify number of images to loan to speed up development
#

import os
import struct
from array import array


class MNISTParser(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self, amount_to_load=-1):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname),
                                amount_to_load)

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self, amount_to_load=-1):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname),
                                amount_to_load)

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl, amount_to_load=-1):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        if amount_to_load == -1:
            for i in range(size):
                images.append([0] * rows * cols)

            for i in range(size):
                images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]
        else:
            for i in range(amount_to_load):
                images.append([0] * rows * cols)

            for i in range(amount_to_load):
                images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

    def print(self, img):
        print(self.display(self.train_images[img]))
