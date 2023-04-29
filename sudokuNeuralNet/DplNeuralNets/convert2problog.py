import json
import random
from typing import Mapping, Iterator
import torch
from problog.logic import Term, Constant, list2term, term2list, Var
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from sudokuNeuralNet.NormalNeuralNets import dataSets
from os import path

def open_dataset(name):
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "../sdata"))
    return json.loads(open(f'{filepath}\\{name}.json', "r").read())

def convert2TermOrConstant(suk, subset):
    size = len(suk)
    subs = dict()
    t = []
    for x in range(size):
        t2 = []
        for y in range(size):
            if suk[x][y] == 'empty':
                t2.append(Term('empty'))
            elif isinstance(suk[x][y], str):
                t2.append(int(suk[x][y]))
            else:
                sub =Term(f'p{x}_{y}')
                subs[sub] = Term('tensor',Term(subset, Constant(suk[x][y])))
                t2.append(subs[sub])
        t2 = list2term(t2)
        t.append(t2)
    return list2term(t)

def convert2VarOrConstant(x):
        return Constant(x)
class MNISTImages(Mapping[Term, torch.Tensor]):

    def __iter__(self) -> Iterator:
        for i in range(self.dataset):
            yield self.dataset[i][0]

    def __len__(self) -> int:
        return len(self.dataset)

    def __init__(self, subset):
        self.subset = subset
        self.dataset = dataSets.mnist[self.subset]

    def __getitem__(self, item):
        return self.dataset[int(item[0])][0]


class SudokuDataset(Dataset):

    def __init__(self, subset, open_file):
        self.subset = "train" #altijd train omdat alle sudoku's train indexen bevat
        self.dataset = open_dataset(open_file)

    def __len__(self):
        return len(self.dataset)

    def get_Sudoku_on_index(self,i):
        return self.dataset[i]
    def to_query(self, i: int) -> Query:    # index i van een training/test voorbeeld
        dic = self.get_Sudoku_on_index(i)
        label = list(dic.keys())[0]
        sudokuAndSolved = list(list(dic.values())[0])
        solved = list2term([list2term([convert2VarOrConstant(x) for x in row]) for row in sudokuAndSolved[2]])
        imagesuk = convert2TermOrConstant(sudokuAndSolved[0], self.subset)

        #term = Term('checkValidSudoku', imagesuk, Var("Out"))
        term = Term('checkValidSudoku', imagesuk)
        p = float(label == 'True')
        return Query(term, p=p)