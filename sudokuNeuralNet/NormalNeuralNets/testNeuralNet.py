from sudokuNeuralNet.NormalNeuralNets.checkIsCorrectSudokuNet import checkIsCorrect
from sudokuNeuralNet.NormalNeuralNets.convertImageSudokuNet import convertImageSudokuNet
from sudokuNeuralNet.NormalNeuralNets.solveSudokuNet import solveSudoku

#net = solveSudoku(lr=1e-04, epoch=10, data_set="train4x4EmptyTrue")
net = convertImageSudokuNet(lr=1e-04, epoch=10, data_set="train4x4NoEmpty")
#net = checkIsCorrect(lr=1e-04, epoch=10, data_set="train4x4Empty")
net.train_model(iter=10000,report=1000)
net.plot()