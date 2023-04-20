import torch
import torch.nn as nn

"""
DPL NET
"""
class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x


"""
Image -> number sudoku (not solved)
"""

class convert_sudoku_image_to_number(nn.Module):
    def __init__(self, sudoku_size,image_size):
        super(convert_sudoku_image_to_number, self).__init__()
        # Unless you pad your image with zeros, a convolutional filter will shrink the size of your output image by filter_size - 1 across the height and width:
        # one conv layer (without padding)	(h, w) -> (h-kernelsize+1, w-kernelsize+1) image 28x28 ->
        # a MaxPool	-> ((h-4)//2, (w-4)//2)
        # self.pool = nn.MaxPool2d(2,2)  # 2x2 max pooling
        #############
        # image layers
        #############
        sq = int(sudoku_size ** (1 / 2))
        # RNN layers
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
            """
        outsize = outsize // 2
        for _ in range(sq):
            cnnLayer.append(nn.Sequential(
                nn.Conv2d(insize, outsize, kernel_size=5, padding=sq),
                # image_size = image_size-kernel_size+1+padding
                nn.ReLU(),
                nn.MaxPool2d(2, 2)))
            insize = outsize
            #"""
        self.cnnLayers = nn.ParameterList(cnnLayer)
        self.liniair = nn.Sequential(
            nn.Linear(outsize*5*5, sudoku_size),
            nn.ReLU())
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        for _, layer in enumerate(self.cnnLayers):
            x = layer(x)
        x = x.view(-1)
        x = self.liniair(x)
        x = self.sigmoid(x)
        return x



"""
Unsolved -> solved sudoku 
"""
class solveNet(nn.Module):
    def __init__(self, sudoku_size):
        super(solveNet, self).__init__()
        sq = int(sudoku_size ** (1/2))
        # CNN layers
        convLayer = []
        insize = 1
        outsize = sudoku_size
        for _ in range(sq):
            convLayer.append(nn.Sequential(
                nn.Conv2d(insize, outsize, kernel_size=sq, padding=sq-1),
                nn.ReLU())
            )
            insize = outsize
            #outsize *= 2
        self.convLayers = nn.ParameterList(convLayer)
        self.liniair_size = outsize*outsize*outsize
        self.out = nn.Linear(self.liniair_size, sudoku_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for _, layer in enumerate(self.convLayers):
            x = layer(x)
        x = x.view(1, self.liniair_size)
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



"""
Checks un_complete sudoku is solved correctly
"""
class Sudoku_Solve_And_Check(nn.Module):
    def __init__(self, sudoku_size):
        super(Sudoku_Solve_And_Check, self).__init__()
        sq = int(sudoku_size ** (1/2))
        # CNN layers
        amount = sq//3
        convLayer = []
        insize = 1
        outsize = sq*sudoku_size
        for _ in range(sq):
            convLayer.append(nn.Sequential(
                nn.Conv2d(insize, outsize, kernel_size=3, padding=2),
                nn.MaxPool2d(2, 2),
                nn.ReLU())
            )
            insize = outsize
        self.convLayers = nn.ParameterList(convLayer)
        self.liniair_size = insize*((sq-(amount*2))**2)
        # stacked liniair layer
        #self.lin = nn.Linear(self.liniair_size, outsize)
        self.solved = nn.Linear(self.liniair_size, sudoku_size)
        linLayer = []
        insize = sudoku_size
        outsize = sudoku_size
        for _ in range(amount):
            linLayer.append(nn.Sequential(
                nn.Linear(insize, outsize),
                nn.ReLU()))
        insize = outsize
        outsize = 1

        for _ in range(amount):
            linLayer.append(nn.Sequential(
                nn.Linear(insize, outsize),
                nn.ReLU()))
        self.linLayers = nn.ParameterList(linLayer)

    def forward(self, x):
        for _, layer in enumerate(self.convLayers):
            x = layer(x)
        x = x.view(1, self.liniair_size)
        x = self.solved(x)
        for _, layer in enumerate(self.linLayers):
            x = layer(x)
        return x


def save_model(model,name):
    torch.save(model.state_dict(), "snapshot/" + name + ".pth")

def load_model(name):
    return torch.load("snapshot/" + name + ".pth")