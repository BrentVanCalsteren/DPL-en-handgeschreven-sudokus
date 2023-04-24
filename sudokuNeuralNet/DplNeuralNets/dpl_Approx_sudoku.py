import torch

from deepproblog.evaluate import get_fact_accuracy
from sudokuNeuralNet.DplNeuralNets.convert2problog import MNISTImages, SudokuDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from sudokuNeuralNet.NormalNeuralNets.torchNet import MNIST_Net

#from sudokuNeuralNet.NormalNeuralNets.torchNet import MNIST_Net

network = MNIST_Net(n=4)
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
pretrain = 1
if pretrain is not None and pretrain > 1:
    network.load_state_dict(torch.load("C:/Users/brent/git/SolveWrittenSudoku/sudokuNeuralNet/DplNeuralNets/snapshot/test_model_ex.pth"))

model = Model("checkvalidsudokuFinalVer.pl", [net])
model.set_engine(ApproximateEngine(model,1, ApproximateEngine.geometric_mean, exploration=False))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

dataset = SudokuDataset("train")
dataset2 = SudokuDataset("test1")

# Train the model
loader = DataLoader(dataset, 1, False)

#opm to self: cross_entropy is default loss function, value tussen 0 en 1
# -> with_negatives moet dan false zijn denk omdat niet negatieve loss krijgen
#possible loss functions "mse", "cross_entropy",
train = train_model(model, loader, 1, log_iter=1, profile=0)
model.save_state("snapshot/test_model.pth")
model.eval()
#get_fact_accuracy(model,dataset2,verbose=2)
#get_fact_accuracy(model,dataset3,verbose=2)
#get_fact_accuracy(model,dataset4,verbose=2)
#get_fact_accuracy(model,dataset5,verbose=2)

# Query the model
query = dataset.to_query(0)
result = model.solve([query])
print(result)