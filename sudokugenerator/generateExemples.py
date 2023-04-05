import random
import sudokuNeuralNet.dataSets as data
import solveSudoku
from sudokugenerator import RandomGenerator as rand


def emptyOrNot(x):
    i = random.randint(0,1)
    if i:
        return x
    else:
        return 0

def generateRandomEmptys(sudoku):
    return [[emptyOrNot(x) for x in row] for row in sudoku]

def generate(amount=100,size=9):
    l = []
    for i in range(amount):
        suk = rand.RandomGenerator.generateRandomSudoku(size)
        suk.matrix = suk.generateRandomEmptys(suk)
        suk2 = suk.getMatrixValues(suk.matrix)
        suk = suk.getMatrixValues(suk.matrix)
        print(solveSudoku.solve(suk2))
        print(suk)
        l.append(suk)
    print(link2MnistFoto(l, size))
    return l


def replaceWithRandomIndex(sudoku, size, subsets):
    a = list()
    for row in sudoku:
        b = list()
        for x in row:
            if x == 0:
                b.append("empty")
            else:
                b.append(random.choice(subsets[x-1]))
        a.append(b)
    return a


def link2MnistFoto(sudokus, size):
    subsets = data.getSubsetMnistlabels("train", size)
    a = list()
    for sudoku in sudokus:
        a.append(replaceWithRandomIndex(sudoku, size, subsets))
    return a

generate(2,9)