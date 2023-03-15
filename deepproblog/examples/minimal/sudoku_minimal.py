import torch


from dataSudoku import MNISTImages, SudokuDataset
from deepproblog.src.deepproblog.dataset import DataLoader
from deepproblog.src.deepproblog.engines import ExactEngine
from deepproblog.src.deepproblog.model import Model
from deepproblog.src.deepproblog.network import Network
from deepproblog.src.deepproblog.train import train_model
from network import MNIST_Net

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("checkValidSudoku.pl", [net])
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

dataset = SudokuDataset("train")

# Train the model
loader = DataLoader(dataset, 2, False)
train_model(model, loader, 1, log_iter=10, profile=0)
model.save_state("snapshot/sudoku_model.pth")

# Query the model
query = dataset.to_query(0)
result = model.solve([query])[0]
print(result)
