#abtract class that every neural net inherence
import json
import random

import torch
from matplotlib import pyplot as plt

from sudokuNeuralNet.NormalNeuralNets import dataSets
from os import path


class neuralNet:
    def __init__(self, epoch, data_set):
        self.dataset = self.open_dataset(data_set)
        _, _, sudoku, _ = self.get_random_data_el()
        self.sqlength = len(sudoku)
        self.epoch = epoch
        self.last_targetValue = None
        self.last_outputValue = None
        self.total_loss = 0
        self.updates = 0
        self.avg_loss_list = list()
        self.report_point_list = list()

    def open_dataset(self, name):
        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath, "..", "sdata"))
        return json.loads(open(f'{filepath}\\{name}.json', "r").read())

    def get_random_data_el(self):
        data = random.choice(self.dataset)
        label = list(data.keys())[0]
        p = float(label == 'True')
        data = list(list(data.values())[0])
        sudoku_foto = self.convert2image(data[0])
        sudoku_unsolved = data[1]
        sudoku_solved = data[2]
        return p, sudoku_foto, sudoku_unsolved, sudoku_solved

    def plot(self, numbers1=[], numbers2=[], label1='iteration', label2='loss'):
        if not numbers1:
            numbers1 = self.report_point_list
        if not numbers2:
            numbers2 = self.avg_loss_list
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

    def rescale(self, sudoku):
        return [(x / self.sqlength) for row in sudoku for x in row]