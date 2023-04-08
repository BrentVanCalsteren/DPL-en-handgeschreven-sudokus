import torchvision
import torchvision.transforms as transforms
import json

"""
Help functions for loading datasets correctly
"""


def getSudokuOnIndexData(subset, index):
    return sudoku_datasets.get(subset)[index]


def getSudokuSetLenght(subset):
    return len(sudoku_datasets.get(subset))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
def getSubsetsMnistlabels(subset, size):
    l = list()
    temp = list()
    for j in range(1, size + 1):
        for i in range(len(mnist[subset])):
            if j == mnist[subset][i][1]:
                temp.append(i)
        l.append(temp)
        temp = list()
    return l

def saveData2json(name, data):
    jsonStr = json.dumps(data)
    jsonFile = open("sudokuNeuralNet/sdata/"+ name + ".json", "w")
    jsonFile.write(jsonStr)
    jsonFile.close()

def open_json(name):
    return json.loads(open(name + ".json", "r").read())

"""
Global DataSets
"""


global mnist; mnist = {
    "train": torchvision.datasets.MNIST(
        root='data/', train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root='data/', train=False, download=True, transform=transform
    ),
}


#global label_indexes; label_indexes = {
#    "train": open_json("sudokuNeuralNet/sdata/trainIndex"),
#}

global sudoku_datasets; sudoku_datasets = {
    "train": open_json("sdata/train4x4WithEmptyTrue"),
    "test": open_json("sdata/stest")
}