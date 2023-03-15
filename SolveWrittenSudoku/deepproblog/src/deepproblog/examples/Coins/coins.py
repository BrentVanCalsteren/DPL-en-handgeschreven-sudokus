import torch

from deepproblog.src.deepproblog.dataset import DataLoader
from deepproblog.src.deepproblog.engines import ExactEngine
from deepproblog.src.deepproblog.evaluate import get_confusion_matrix
from deepproblog.src.deepproblog.examples.Coins.data.dataset import train_dataset, test_dataset
from deepproblog.src.deepproblog.model import Model
from deepproblog.src.deepproblog.network import Network
from deepproblog.src.deepproblog.train import train_model
from deepproblog.src.deepproblog.utils.standard_networks import smallnet
from deepproblog.src.deepproblog.utils.stop_condition import Threshold, StopOnPlateau

batch_size = 5
loader = DataLoader(train_dataset, batch_size)
lr = 1e-4
coin_network1 = smallnet(num_classes=2, pretrained=True)
coin_net1 = Network(coin_network1, "net1", batching=True)
coin_net1.optimizer = torch.optim.Adam(coin_network1.parameters(), lr=lr)
coin_network2 = smallnet(num_classes=2, pretrained=True)
coin_net2 = Network(coin_network2, "net2", batching=True)
coin_net2.optimizer = torch.optim.Adam(coin_network2.parameters(), lr=lr)

model = Model("model.pl", [coin_net1, coin_net2])
model.add_tensor_source("train", train_dataset)
model.add_tensor_source("test", test_dataset)
model.set_engine(ExactEngine(model), cache=True)
train_obj = train_model(
    model,
    loader,
    StopOnPlateau("Accuracy", warm_up=10, patience=10)
    | Threshold("Accuracy", 1.0, duration=2),
    log_iter=100 // batch_size,
    test_iter=100 // batch_size,
    test=lambda x: [("Accuracy", get_confusion_matrix(x, test_dataset).accuracy())],
    infoloss=0.25,
)
