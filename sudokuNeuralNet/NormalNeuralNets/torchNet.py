import json

import torch
import torch.nn as nn
from pathlib import Path

"""
DPL NET
"""
class MNIST_Net(nn.Module):
    def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
        super(MNIST_Net, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


"""
Image -> number sudoku (not solved)
"""
class oneHugeNet2(nn.Module):
    def __init__(self, sudoku_size):
        super(oneHugeNet2, self).__init__()
        self.mnistNet = MNIST_Net(int(sudoku_size**(1/2)))
        self.sudokusize = sudoku_size
        self.logicNet = Sudoku_Checker(sudoku_size)

    def forward(self, x):
        xList = torch.split(x, 1)
        outputFirstNet = [self.mnistNet(a) for a in xList]
        outputFirstNet = [convert(a,self.sudokusize**(1/2)) for a in outputFirstNet]
        input = torch.cat(outputFirstNet,dim=0)
        x = self.logicNet(input)
        return x


def convert(tensor,scale):
    index = torch.argmax(tensor, dim=1).item()
    t = tensor[0][index]
    t = (t/t*index+1)/scale
    return t.unsqueeze(0)

class oneHugeNet(nn.Module):
    def __init__(self, sudoku_size):
        super(oneHugeNet, self).__init__()
        # Unless you pad your image with zeros, a convolutional filter will shrink the size of your output image by filter_size - 1 across the height and width:
        # one conv layer (without padding)	(h, w) -> (h-kernelsize+1, w-kernelsize+1) image 28x28 ->
        # a MaxPool	-> ((h-4)//2, (w-4)//2)
        # self.pool = nn.MaxPool2d(2,2)  # 2x2 max pooling
        #############
        # image layers
        #############
        sq = int(sudoku_size ** (1 / 2))
        # CNN layers
        cnnLayer = []
        insize = sudoku_size
        outsize = sudoku_size * 2
        for _ in range(sq):
            cnnLayer.append(nn.Sequential(
            nn.Conv2d(insize, outsize, kernel_size=5, padding=sq),
            # image_size = image_size-kernel_size+1+padding
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # pak grootste waarde 2x2 window image_size = image_size/2 | 24/2 = 12
        ))
            insize = outsize

        insize = insize*5*5
        outsize = insize // 2
        self.cnnLayers = nn.ParameterList(cnnLayer)
        #RNN layers
        rnnLayer = []
        for _ in range(sq):
            rnnLayer.append(nn.Sequential(
            nn.Linear(insize, outsize),
            # image_size = image_size-kernel_size+1+padding
            nn.ReLU()
            ))
            insize = outsize
            outsize = insize // 2
        self.rnnLayers = nn.ParameterList(rnnLayer)
        self.out = nn.Sequential(
            nn.Linear(insize, 1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        for _, layer in enumerate(self.cnnLayers):
            x = layer(x)
        x = x.view(-1)
        for _, layer in enumerate(self.rnnLayers):
            x = layer(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


"""
Checks complete sudoku is solved correctly
"""
class Sudoku_Checker(nn.Module):
    def __init__(self, sudoku_size):
        super(Sudoku_Checker, self).__init__()
        sq = int(sudoku_size ** (1/2))
        # RNN layers
        linLayer = []
        insize = sudoku_size
        outsize = sudoku_size*2
        for _ in range(sq):
            linLayer.append(nn.Sequential(
                nn.Linear(insize, outsize),
                nn.ReLU()))
            insize = outsize
        outsize = sudoku_size//2
        for _ in range(sq):
            linLayer.append(nn.Sequential(
                nn.Linear(insize, outsize),
                nn.ReLU()))
            insize = outsize
        self.linLayers = nn.ParameterList(linLayer)
        outsize = 1
        self.out = nn.Linear(insize, outsize)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for _, layer in enumerate(self.linLayers):
            x = layer(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x



def save_model(model,name):
    torch.save({
        'model_state_dict': model.state_dict()},
        f'snapshot/{name}.pth')

def load_in_model(model,name):
    checkpoint = torch.load(f'snapshot/{name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

def saveTrainData2json(data, name):
    jsonStr = json.dumps(data)
    jsonFile = open("data/" + name + ".json", "w")
    jsonFile.write(jsonStr)
    jsonFile.close()