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
import time


network = MNIST_Net(n=4)
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("checkvalidsudokuNoPref.pl", [net])
model.set_engine(ExactEngine(model),sdd_auto_gc=True)
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

def train_net():
    for i in range(eval_itertion):
        # load dataset
        trainset = dataset_name + f'_eval{i}'
        dataset = SudokuDataset("train", trainset)
        loader = DataLoader(dataset, 1, False)
        # Train the model
        logger = train_model(model, loader, 1, loss_function_name="cross_entropy", log_iter=1).logger
        data = logger.log_data
        # eval the model
        model.eval()
        accu_matrix = get_fact_accuracy(model, testdata, verbose=2).accuracy()
        # save results
        save_model(network, savefile_name)
        load_in_model(eval_on_single_image.model, savefile_name)
        image_acc = eval_on_single_image.get_accuracy(dset=eval_on_single_image.dataset)
        print("Image accuricy: " + str(image_acc))
        data_list.append([data['time'], data['loss'], accu_matrix, image_acc])

    saveTrainData2json(data_list, savefile_name)

def solve_querries():
    for i in range(eval_itertion):
        # load dataset
        trainset = dataset_name + f'_eval{i}'
        dataset = SudokuDataset("train", trainset)
        # eval the model
        model.eval()
        start = time.time()
        for i, query in enumerate(dataset.to_queries()):
            if i > 4: #hoeveel queries ge wilt checken
                break
            result = model.solve([query])
            print(result[0])
            querry_time = time.time()-start
            print("Time: " + str(querry_time))
            data_list.append(querry_time)

    saveTrainData2json(data_list, savefile_name)

def check_circle_permutation():
    image_acc1 = eval_on_single_image.get_accuracy(dset=eval_on_single_image.dataset,offset=0)
    image_acc2 = eval_on_single_image.get_accuracy(dset=eval_on_single_image.dataset,offset=1)
    image_acc3 = eval_on_single_image.get_accuracy(dset=eval_on_single_image.dataset,offset=2)
    image_acc4 = eval_on_single_image.get_accuracy(dset=eval_on_single_image.dataset,offset=3)
    saveTrainData2json([image_acc1,image_acc2,image_acc3,image_acc4], f"offset_acc{images}_part2")

eval_itertion = 5
images = 2
data_list = list()
testdata = SudokuDataset("test","test4x4_50sudokus")
eval_on_single_image = mnistNet(lr=1e-04, epoch=1, data_set='test4x4_All_Images')
dataset_name = f'train4x4_50sudokus_{images}images'
savefile_name = f"train4x4_{images}images_part2"
load_in_model(eval_on_single_image.model,f"train4x4_{images}images_part2")
train_net()









