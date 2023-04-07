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
        i = random.randint(0, 1)
        if i:
            suk.matrix = suk.makeRandomInvalid(suk, size)
        suk2 = suk.getMatrixValues(suk.matrix)
        suk = suk.getMatrixValues(suk.matrix)
        label = solveSudoku.solve(suk2)
        print(label)
        print(suk)
        indexed = link2MnistFoto(suk)
        combo = {str(label): [indexed, suk]}
        l.append(combo)
    print(l)
    data.saveData2json("train4x4WithEmpty",l)
    return l


def replaceWithRandomIndex(sudoku, subsets):
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


def link2MnistFoto(sudoku):
    subsets = data.label_indexes["train"]
    return replaceWithRandomIndex(sudoku, subsets)

generate(10000,4)