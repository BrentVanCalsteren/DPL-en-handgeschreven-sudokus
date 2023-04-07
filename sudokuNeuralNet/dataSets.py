import torchvision
import torchvision.transforms as transforms
import json

"""
Help functions for loading datasets correctly
"""

global vars; vars = ['A','B','C','D','E','F','G','H','I',
                     'J','K','L','M','N','O','P','Q','R',
                     'S','T','U','V','W','X','Y','Z','AA',
                     'AB','AC','AD','AE','AF','AG','AH','AI','AJ',
                     'AK','AL','AM','AN','AO','AP','AQ','AR','AS',
                     'AT','AU','AV','AW','AX','AY','AZ','BA','BB',
                     'BC','BD','BE','BF','BG','BH','BI','BJ','BK',
                     'BL','BM','BN','BO','BP','BQ','BR','BS','BT',
                     'BU','BV','BW','BX','BY','BZ','CA','CB','CC',
                     'CD','CE','CF','CG','CH','CI','CJ','CK','CL',
                     ]


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
def getSubsetsMnistlabels(subset, size):
    l = list()
    temp = list()
    for j in range(1, size + 1):
        for i in range(len(datasets[subset])):
            if j == datasets[subset][i][1]:
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


global datasets; datasets = {
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
    "train": open_json("sdata/train4x4WithEmpty"),
    "test": open_json("sdata/stest")
}