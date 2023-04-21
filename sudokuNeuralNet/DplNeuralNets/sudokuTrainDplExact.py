import torch


from sudokuNeuralNet.DplNeuralNets.convert2problog import MNISTImages, SudokuDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_fact_accuracy
from sudokuNeuralNet.NormalNeuralNets.torchNet import MNIST_Net

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("checkValidSudokuVer4slow.pl", [net])
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

dataset = SudokuDataset("train")
dataset2 = SudokuDataset("test1")
dataset3 = SudokuDataset("test2")
dataset4 = SudokuDataset("test3")
dataset5 = SudokuDataset("test4")

# Train the model
loader = DataLoader(dataset, 1, True)

#opm to self: cross_entropy is default loss function, value tussen 0 en 1
# -> with_negatives moet dan false zijn denk omdat niet negatieve loss krijgen
#possible loss functions "mse", "cross_entropy",
train_model(model, loader, 1, loss_function_name="cross_entropy", log_iter=1)
model.save_state("snapshot/sudoku_model.pth")
# Query the model
query = dataset.to_query(2)
model.eval()
#get_fact_accuracy(model,dataset2,verbose=2)
#get_fact_accuracy(model,dataset3,verbose=2)
#get_fact_accuracy(model,dataset4,verbose=2)
get_fact_accuracy(model,dataset5,verbose=2)
result = model.solve([query])[0]

print(result, result.result.values())
