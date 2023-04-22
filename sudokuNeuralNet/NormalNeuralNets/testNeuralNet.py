from sudokuNeuralNet.NormalNeuralNets import torchNet
from sudokuNeuralNet.NormalNeuralNets.IsSudokuCorrectNet import isCorrectNet
from sudokuNeuralNet.NormalNeuralNets.oneHugeNet import oneHugeNet
from sudokuNeuralNet.NormalNeuralNets.mnistNet import mnistNet
from sudokuNeuralNet.NormalNeuralNets.solveSudokuNet import solveSudoku

#net = isCorrectNet(lr=1e-04, epoch=10, data_set="train4x4EmptyImage")
#net = oneHugeNet(lr=1e-04, epoch=10, data_set="train4x4Image")
net = mnistNet(lr=1e-04, epoch=1,data_set="train4x416Image")
net.train_model(iter=50000,report=100)
net.saveTrainData2json("mnistTrainData1")
torchNet.save_model(net.model,"test")
net.plot()