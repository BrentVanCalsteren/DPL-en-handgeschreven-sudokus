import torch


from sudokuNeuralNet.DplNeuralNets.convert2problog import MNISTImages, SudokuDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from sudokuNeuralNet.NormalNeuralNets.torchNet import MNIST_Net

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("checkValidSudokuVer4.pl", [net])
model.set_engine(ApproximateEngine(model,1, ApproximateEngine.geometric_mean, exploration=True))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

dataset = SudokuDataset("train")

# Train the model
loader = DataLoader(dataset, 2, False)

#opm to self: cross_entropy is default loss function, value tussen 0 en 1
# -> with_negatives moet dan false zijn denk omdat niet negatieve loss krijgen
#possible loss functions "mse", "cross_entropy",
train = train_model(model, loader, 1, log_iter=1, profile=0)
model.save_state("snapshot/sudoku_model.pth")

# Query the model
query = dataset.to_query(0)
result = model.solve([query])[0]
print(result.result)