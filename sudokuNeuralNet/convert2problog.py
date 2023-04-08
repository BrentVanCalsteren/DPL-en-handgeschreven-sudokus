from typing import Mapping, Iterator
import torch
from problog.logic import Term, Constant, list2term
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from sudokuNeuralNet import dataSets


def convert2TermOrConstant(suk, subset):
    t = []
    for x in range(4):
        t2 = []
        for y in range(4):
            if suk[x][y] == 'empty':
                t2.append(Constant('empty'))
            else:
                t2.append(Term('tensor', Term(subset, Constant(suk[x][y]))))
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

    def __init__(self, subset):
        self.subset = subset
        self.dataset = dataSets.sudoku_datasets[subset]

    def __len__(self):
        return len(self.dataset)

    def to_query(self, i: int) -> Query:    # index i van een training/test voorbeeld
        dic = dataSets.getSudokuOnIndexData(self.subset, i)
        label = list(dic.keys())[0]
        sudokuAndSolved = list(list(dic.values())[0])
        solved = list2term([list2term([convert2VarOrConstant(x) for x in row]) for row in sudokuAndSolved[1]])
        sudoku = convert2TermOrConstant(sudokuAndSolved[0], self.subset)
        term = Term('checkValidSudoku', sudoku)
        p = float(label == 'True')
        return Query(term, p=p)
