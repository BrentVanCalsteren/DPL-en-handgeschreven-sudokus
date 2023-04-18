#abtract class that every neural net inherence
import json
import random

import torch
from matplotlib import pyplot as plt

from sudokuNeuralNet.NormalNeuralNets import dataSets


class neuralNet:


    def __int__(self):
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.last_targetValue = None
        self.last_outputValue = None
        self.iterations = 0
        self.epoch = 0
        self.avg_loss_each_epoch = list()
        self.criterion = None

    def train_model(self):
       pass

    def forward(self, input):
        pass

    def get_loss(self, target):
        pass

    def update_net(self, loss):
        pass

    def open_dataset(self, name):
        return json.loads(open("sdata/" + name + ".json", "r").read())

    def get_random_data_el(self):
        data = random.choice(self.dataset)
        label = list(data.keys())[0]
        p = float(label == 'True')
        data = list(list(data.values())[0])
        sudoku_foto = self.convert2image(data[0])
        sudoku_unsolved = data[1]
        sudoku_solved = data[2]
        return p, sudoku_foto, sudoku_unsolved, sudoku_solved

    def plot(self, numbers1=[], numbers2=[], label1='', label2=''):
        plt.plot(numbers1, numbers2)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.show()

    def convert2image(self, sudoku):
        list = [self.load_image(x) for row in sudoku for x in row]
        return torch.stack(list, dim=1)
        # tuples = [torch.squeeze(torch.tensor(el)).tolist() for el in list]
        # return tuples

    def load_image(self, index):
        if index == 'empty':
            return torch.zeros((1, 28, 28))
        else:
            image, label = dataSets.mnist["train"][index]
            return image

    def rescale(sudoku):
        l = len(sudoku)
        return [(x / l) for row in sudoku for x in row]