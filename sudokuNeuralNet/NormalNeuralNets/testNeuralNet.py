import torch

from sudokuNeuralNet.NormalNeuralNets import torchNet
from sudokuNeuralNet.NormalNeuralNets.IsSudokuCorrectNet import isCorrectNet
from sudokuNeuralNet.NormalNeuralNets.oneHugeNet import oneHugeNet
from sudokuNeuralNet.NormalNeuralNets.mnistNet import mnistNet
from sudokuNeuralNet.NormalNeuralNets.solveSudokuNet import solveSudoku

#net = isCorrectNet(lr=1e-04, epoch=10, data_set="train4x4EmptyImage")
#net = oneHugeNet(lr=1e-4, epoch=1, data_set="train4x4Image")


net = mnistNet(lr=1e-03, epoch=1,data_set="train9x9_81Image")

print(net.get_accuracy())
net.train_model(iter=300, report=100)
#net.saveTrainData2json("mnistTrainData1")
net.saveTrainData2json("9_MNIST_net_data")
torchNet.save_model(net.model,"9_mnist95+")
net.plot()