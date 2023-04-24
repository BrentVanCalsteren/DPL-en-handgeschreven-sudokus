import json

import torch


from sudokuNeuralNet.DplNeuralNets.convert2problog import MNISTImages, SudokuDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_fact_accuracy
from sudokuNeuralNet.NormalNeuralNets.mnistNet import mnistNet
from sudokuNeuralNet.NormalNeuralNets.torchNet import MNIST_Net, load_in_model,save_model,saveTrainData2json

network = MNIST_Net(n=4)
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("checkvalidsudokuFinalVer.pl", [net])
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))


eval_itertion = 5
images = 2
data_list = list()
testdata = SudokuDataset("test","test100exemples6images")
eval_on_single_image = mnistNet(lr=1e-04, epoch=1,data_set="train4x416Image")
overal_name = f'train4x4ImagesTrueAllEmpty{images}'

for i in range(eval_itertion):
    #load dataset
    trainset = overal_name + f'eval{i}'
    dataset = SudokuDataset("train", trainset)
    loader = DataLoader(dataset, 1, False)
    # Train the model
    logger = train_model(model, loader, 1, loss_function_name="cross_entropy", log_iter=1).logger
    data = logger.log_data
    # eval the model
    model.eval()
    accu_matrix = get_fact_accuracy(model, testdata, verbose=2).accuracy()
    #save results
    save_model(network,overal_name)
    load_in_model(eval_on_single_image.model, overal_name)
    image_acc = eval_on_single_image.get_accuracy()
    print("Image accuricy: " + str(image_acc))
    data_list.append([data['time'], data['loss'], accu_matrix, image_acc])

saveTrainData2json(data_list, overal_name)






