from sudokuNeuralNet.NormalNeuralNets.IsSudokuCorrectNet import isCorrectNet
from sudokuNeuralNet.NormalNeuralNets.convertImageSudokuNet import convertImageSudokuNet
from sudokuNeuralNet.NormalNeuralNets.mnistNet import mnistNet
from sudokuNeuralNet.NormalNeuralNets.solveSudokuNet import solveSudoku

net = isCorrectNet(lr=1e-04, epoch=10, data_set="train4x4EmptyImage")
#net = convertImageSudokuNet(lr=1e-04, epoch=10, data_set="train4x4NoEmpty")
#net = checkIsCorrect(lr=1e-04, epoch=10, data_set="train4x4Empty")
#net = mnistNet(lr=1e-03, epoch=1,data_set="train4x416Image")
net.train_model(iter=10000,report=1000)
net.plot()