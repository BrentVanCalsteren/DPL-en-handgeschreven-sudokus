import random
import dataSets as data
import solveSudoku
from sudokuNeuralNet.NormalNeuralNets import neuralNet
from sudokugenerator import RandomGenerator as rand



def main():
    size = 4
    eval_itertion = 10
    images = 2
    empty = 15
    #generate(amount=100, size=4, convert_number=6, onlyTrue=False, withEmptys=False, name='test100exemples6images')
    #"""
    for i in range(eval_itertion):
        trainset = f'train4x4ImagesTrueAllEmpty{images}eval{i}'
        generate(amount=100, size=size, convert_number=images,emptys=empty, onlyTrue=True, withEmptys=True, name=trainset)
    """
    #"""
def generate(amount=100,size=9, name="temp", onlyTrue=False,withEmptys=True, convert_number=None, emptys=-1):
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
        indexed = link2MnistFoto(suk, convert_number)
        if convert_number != None and emptys >= convert_number:
            removeNumbers(indexed)
        combo = {str(label): [indexed, suk, suk2]}
        l.append(combo)
    data.saveData2json(name, l)
    return l

def removeNumbers(sudoku):
    size = len(sudoku)
    for i in range(size):
        for j in range(size):
            if isinstance(sudoku[i][j],str):
                sudoku[i][j] = "empty"

def replaceWithRandomIndex(sudoku, subsets, convertNumber=None):
    size = len(sudoku)**2
    if convertNumber:
        nonconvert = size - convertNumber
    else:
        nonconvert = size
    a = list()
    for row in sudoku:
        b = list()
        for x in row:
            if x == 0:
                b.append("empty")
                nonconvert-=1
            else:
                if not convertNumber == None:
                    if convertNumber > 0:
                        b.append(random.choice(subsets[x - 1]))
                        convertNumber -= 1
                    elif nonconvert > 0:
                        r = random.randint(0, nonconvert)
                        if r or convertNumber <= 0:
                            b.append(str(x))
                            nonconvert -= 1
                        else:
                            b.append(random.choice(subsets[x - 1]))
                            convertNumber -= 1
                    else:
                        b.append(str(x))
                else:
                    b.append(random.choice(subsets[x - 1]))

        a.append(b)
    return a


def link2MnistFoto(sudoku, convertNumber=None):
    subsets = data.label_indexes["train"]
    return replaceWithRandomIndex(sudoku, subsets,convertNumber)

def flip_dataset(datset,name):
    flipped_set = list()

    set = neuralNet.open_dataset(datset)
    for s in set:
        a = dict()
        b = list(s.values())[0]
        c = [[flip_string(x) for x in row] for row in b[0]]
        d = [[flip_values(x) for x in row] for row in b[1]]
        e = [[flip_values(x) for x in row] for row in b[2]]
        b = [c,d,e]
        a[list(s.keys())[0]] = b
        flipped_set.append(a)
    data.saveData2json(name,flipped_set)
def flip_string(l):
    if l == "1":
        return "2"
    if l == "2":
        return "3"
    if l == "3":
        return "4"
    if l == "4":
        return "1"
    return l

def flip_values(l):
    if l == 1:
        return 2
    if l == 2:
        return 3
    if l == 3:
        return 4
    if l == 4:
        return 1
    return l


if __name__ == '__main__':
    main()
