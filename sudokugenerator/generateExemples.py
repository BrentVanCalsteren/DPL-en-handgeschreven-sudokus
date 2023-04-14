import random
import dataSets as data
import solveSudoku
from sudokugenerator import RandomGenerator as rand


def main():
    generate(amount=10000, size=9, name="train9x9NoEmpty",onlyTrue=False,withEmptys=False)

def generate(amount=100,size=9, name="temp", onlyTrue=False,withEmptys=True):
    l = []
    for i in range(amount):
        suk = rand.RandomGenerator.generateRandomSudoku(size)
        if withEmptys:
            suk.matrix = suk.generateRandomEmptys(suk)
        if not onlyTrue:
            if withEmptys:
                suk.matrix = suk.makeRandomInvalid(suk, size)
            else:
                a = random.randint(0,1)
                if a:
                    suk.matrix = suk.makeRandomInvalid(suk, size)
        suk2 = suk.getMatrixValues(suk.matrix)
        suk = suk.getMatrixValues(suk.matrix)
        label = solveSudoku.solve(suk2)
        indexed = link2MnistFoto(suk)
        combo = {str(label): [indexed, suk, suk2]}
        l.append(combo)
    data.saveData2json(name, l)
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


if __name__ == '__main__':
    main()