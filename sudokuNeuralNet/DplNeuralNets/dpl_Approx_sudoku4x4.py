import torch

from deepproblog.evaluate import get_fact_accuracy
from sudokuNeuralNet.DplNeuralNets.convert2problog import MNISTImages, SudokuDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from sudokuNeuralNet.NormalNeuralNets.mnistNet import mnistNet
from sudokuNeuralNet.NormalNeuralNets.torchNet import MNIST_Net, load_in_model,save_model,saveTrainData2json
import time


network = MNIST_Net(n=4)
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
model = Model("checkvalidsudokuNoPref.pl", [net])
#set engine: heuristic opties zijn ApproximateEngine.geometric_mean / ApproximateEngine.ucs
#exploration kan ervoor zorgen sneller leren, maar langer duren voor vinden oplossing
model.set_engine(ApproximateEngine(model,1, ApproximateEngine.geometric_mean, exploration=False))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))
#voor werken met pretrained net
# network.load_state_dict(torch.load("C:/Users/brent/git/SolveWrittenSudoku/sudokuNeuralNet/DplNeuralNets/snapshot/test_model_ex.pth"))

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


eval_itertion = 5
images = 0
data_list = list()
testdata = SudokuDataset("test","test4x4_50sudokus")
eval_on_single_image = mnistNet(lr=1e-04, epoch=1, data_set='test4x4_All_Images')
dataset_name = f"train4x4_50sudokus_with_numbers&NoNegs"
savefile_name = f"train4x4_with_NoNegs"
train_net()