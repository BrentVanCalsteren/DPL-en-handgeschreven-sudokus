import copy
import random
import dataSets as data
import solveSudoku
from sudokuNeuralNet.NormalNeuralNets import neuralNet
from sudokugenerator import RandomGenerator as rand



def main():
    #create permutation -> schuif value 1 naar rechts
    """
    dat = data.open_json('test4x4_100sudokus_with_numbers_perm3')
    permu_dataset(dat,'test4x4_100sudokus_with_numbers_perm4')
    """#"""
    #"""
    size = 4
    eval_itertion = 5
    images = 5
    empty = 5
    for i in range(eval_itertion):
        trainset = f'train4x4_50sudokus_with_numbers&NoNegs_eval{i}'
        generate(amount=50, size=size, convert_number=images,emptys=empty, onlyTrue=True, withEmptys=True, name=trainset)
    """#"""

def generate(amount=10,size=9, name="temp", onlyTrue=False,withEmptys=True, convert_number=None, emptys=-1):
    l = []
    for i in range(amount):
        suk = rand.RandomGenerator.generateRandomSudoku(size)
        if withEmptys:
            suk.matrix = suk.generateRandomEmptys(suk,emptys)
        if not onlyTrue:
            a = random.randint(0,1)
            if a:
                suk.matrix = suk.makeRandomInvalid(suk, size)
        suk2 = suk.getMatrixValues(suk.matrix)
        suk = suk.getMatrixValues(suk.matrix)
        label = solveSudoku.solve(suk2)
        indexed = link2MnistFoto(suk, convert_number, emptys)
        combo = {str(label): [indexed, suk, suk2]}
        l.append(combo)
    data.saveData2json(name, l)
    return l

def replaceWithRandomIndex(sudoku, subsets, convertNumber=0, emptys=0):
    if convertNumber + emptys > len(sudoku)**2:
        return []
    suk = copy.deepcopy(sudoku)
    size = len(sudoku)
    replace_Zeros_convert(suk, size)
    while convertNumber:
        k = random.randint(0,size-1)
        l = random.randint(0,size-1)
        index = suk[k][l]
        if index != "empty" and isinstance(index, str):
            suk[k][l] = random.choice(subsets[int(index) - 1])
            convertNumber -= 1
    return suk

def replace_Zeros_convert(suk,size):
    for x in range(size):
        for y in range(size):
            index = suk[x][y]
            if index == 0:
                suk[x][y] = "empty"
            else:
                suk[x][y] = str(index)

def removeNumbers(sudoku):
    size = len(sudoku)
    for i in range(size):
        for j in range(size):
            if isinstance(sudoku[i][j],str):
                sudoku[i][j] = "empty"

def link2MnistFoto(sudoku, convertNumber=0, emptys=0):
    subsets = data.label_indexes["train"]
    return replaceWithRandomIndex(sudoku, subsets,convertNumber,emptys)

def permu_dataset(datset,name):
    flipped_set = list()

    size = len(list(datset[0].values())[0][0])

    if size == 4:
        for s in datset:
            a = dict()
            b = list(s.values())[0]
            c = [[flip_string4x4(x) for x in row] for row in b[0]]
            d = [[flip_values4x4(x) for x in row] for row in b[1]]
            e = [[flip_values4x4(x) for x in row] for row in b[2]]
            b = [c,d,e]
            a[list(s.keys())[0]] = b
            flipped_set.append(a)
    elif size == 9:
        for s in set:
            a = dict()
            b = list(s.values())[0]
            c = [[flip_string4x4(x) for x in row] for row in b[0]]
            d = [[flip_values4x4(x) for x in row] for row in b[1]]
            e = [[flip_values4x4(x) for x in row] for row in b[2]]
            b = [c,d,e]
            a[list(s.keys())[0]] = b
            flipped_set.append(a)
    data.saveData2json(name,flipped_set)
def flip_string4x4(l):
    if l == "1":
        return "2"
    if l == "2":
        return "3"
    if l == "3":
        return "4"
    if l == "4":
        return "1"
    return l

def flip_values4x4(l):
    if l == 1:
        return 2
    if l == 2:
        return 3
    if l == 3:
        return 4
    if l == 4:
        return 1
    return l


def flip_string9x9(l):
    if l == "1":
        return "2"
    elif l == "2":
        return "3"
    elif l == "3":
        return "4"
    elif l == "4":
        return "5"
    elif l == "5":
        return "6"
    elif l == "6":
        return "7"
    elif l == "7":
        return "8"
    elif l == "8":
        return "9"
    elif l == "9":
        return "1"
    return l

def flip_values9x9(l):
    if l == 1:
        return 2
    elif l == 2:
        return 3
    elif l == 3:
        return 4
    elif l == 4:
        return 5
    elif l == 5:
        return 6
    elif l == 6:
        return 7
    elif l == 7:
        return 8
    elif l == 8:
        return 9
    elif l == 9:
        return 1
    return l


if __name__ == '__main__':
    main()
