import torch

from sudokuNeuralNet.NormalNeuralNets import torchNet
from sudokuNeuralNet.NormalNeuralNets.IsSudokuCorrectNet import isCorrectNet
from sudokuNeuralNet.NormalNeuralNets.oneHugeNet import oneHugeNet
from sudokuNeuralNet.NormalNeuralNets.mnistNet import mnistNet
from sudokuNeuralNet.NormalNeuralNets.solveSudokuNet import solveSudoku

#net = isCorrectNet(lr=1e-04, epoch=10, data_set="train4x4EmptyImage")
#net = oneHugeNet(lr=1e-4, epoch=1, data_set="train4x4Image")
network = torchNet.MNIST_Net(n=4)
a = torch.load(
    "C:/Users/brent/git/SolveWrittenSudoku/sudokuNeuralNet/DplNeuralNets/snapshot/test_model_ex.pth")


net = mnistNet(lr=1e-04, epoch=1,data_set="train4x416Image")

net.load_pretrained_model("test_model_ex")
print(net.get_accuracy())
#net.train_model(iter=50000,report=100)
#net.saveTrainData2json("mnistTrainData1")
net.saveTrainData2json("hugeNetTrainData1")
torchNet.save_model(net.model,"hugeNet50000")
net.plot()