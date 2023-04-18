import torch

from sudokuNeuralNet.NormalNeuralNets import torchNet, dataSets
from sudokuNeuralNet.NormalNeuralNets.neuralNet import neuralNet


class checkIsSolved(neuralNet):
    def __int__(self, data_set, lr, epoch):
        self.dataset = self.open_dataset(data_set)
        _, sudoku, _, _ = self.get_random_data_el()
        self.length = len(sudoku)
        self.model = torchNet.Sudoku_Checker(sudoku_size=self.length)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.last_targetValue = None
        self.last_outputValue = None
        self.last_loss = None
        self.updates = 0
        self.epoch = epoch
        self.avg_loss_each_epoch = list()
        self.criterion = torch.nn.BCELoss()


    def train_model(self, iter):
        for e in range(self.epoch):
            for i in range(iter):
                target, _, input, _ = self.get_random_data_el()
                self.forward(input)
                loss = self.get_loss(target)
                self.update_net(loss)


    def forward(self, input):
        sudoku = self.rescale(input)
        torch_input = torch.tensor(sudoku, dtype=torch.float32).view(1, self.length ** 2)
        self.last_outputValue = self.model(torch_input)
        return self.last_outputValue

    def get_loss(self, target):
        sudoku = self.rescale(target)
        self.last_targetValue = torch.tensor(sudoku, dtype=torch.float32).view(1, self.length ** 2)
        return self.criterion(self.last_outputValue, self.last_targetValue)


    def update_net(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()