import random
import json
from typing import Mapping, Iterator
import ast
import torch
import torchvision
import torchvision.transforms as transforms
from problog.logic import Term, Constant, list2term

from torchvision.datasets import MNIST

from deepproblog.dataset import Dataset
from deepproblog.query import Query

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root='data/', train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root='data/', train=False, download=True, transform=transform
    ),
}
def getMnistlabels(subset):
    l = list()
    l1 = list()
    l2 = list()
    l3 = list()
    l4 = list()
    for i in range(len(datasets[subset])):
        match datasets[subset][i][1]:
            case 1:
                l1.append(i)
            case 2:
                l2.append(i)
            case 3:
                l3.append(i)
            case 4:
                l4.append(i)
    l.append(l1)
    l.append(l2)
    l.append(l3)
    l.append(l4)
    return l

labelsIndex = {
    "sdata/strain.csv": getMnistlabels("train"),
    "sdata/stest.csv": getMnistlabels("test")
}

def convert2randomIndex(suk2, allLabels):
    a = list()
    for suk in suk2:
        b = list()
        for x in suk:
            match x:
                case 1:
                    b.append(random.choice(allLabels[0]))
                case 2:
                    b.append(random.choice(allLabels[1]))
                case 3:
                    b.append(random.choice(allLabels[2]))
                case 4:
                    b.append(random.choice(allLabels[3]))
        a.append(b)
    return a
def open_json(name):
    jsonFile = open(name + ".json", "r")
    jsonContent = jsonFile.read()
    return json.loads(jsonContent)

#method for converting and processing cvg 2 json
def opencvg_convert2json(name):
    l = list()
    allLabels = labelsIndex[name + ".csv"]
    with open(name + ".csv") as file:
        for line in file:
            suk, label = str(line).strip().split(".")
            suk = ast.literal_eval(suk)
            suk2 = convert2randomIndex(suk, allLabels)
            combo = {str(label): [suk2,suk]}
            l.append(combo)
    jsonStr = json.dumps(l)
    jsonFile = open(name + ".json", "w")
    jsonFile.write(jsonStr)
    jsonFile.close()
    return l

sudoku_datasets = {
    "train": open_json("sdata/strain"),
    "test": open_json("sdata/stest")
}


def getSudokuOnIndexData(subset, index):
    return sudoku_datasets.get(subset)[index]

def getSudokuOnIndexLabel(subset, index):
    return sudoku_datasets[subset][index].split('.')[1]

def convert2Term(suk, subset):
    t = []
    t2 = []
    for x in range(4):
        for y in range(4):
            t2.append(Term('tensor', Term(subset, Constant(suk[x][y]))))
        t2 = list2term(t2)
        t.append(t2)
        t2 = []
    return list2term(t)

class MNISTImages(Mapping[Term, torch.Tensor]):

    def __iter__(self) -> Iterator:
        for i in range(self.dataset):
            yield self.dataset[i][0]

    def __len__(self) -> int:
        return len(self.dataset)

    def __init__(self, subset):
        self.subset = subset
        self.dataset = datasets[self.subset]

    def __getitem__(self, item):
        return self.dataset[int(item[0])][0]


class SudokuDataset(Dataset):

    def __init__(self, subset):
        self.subset = subset
        self.dataset = sudoku_datasets[subset]

    def __len__(self):
        return len(self.dataset)

    def to_query(self, i: int) -> Query:    # index i van een training/test voorbeeld
        dic = getSudokuOnIndexData(self.subset, i)
        label = list(dic.keys())[0]
        sudokuAndSolved = list(list(dic.values())[0])
        solved = list2term([list2term([Constant(x) for x in row]) for row in sudokuAndSolved[1]])
        sudoku = convert2Term(sudokuAndSolved[0], self.subset)
        term = Term('checkValidSudoku', sudoku, solved)
        # label = getSudokuOnIndexLabel(self.subset, i)
        p = float(label == 'True')
        return Query(term, p=p)
