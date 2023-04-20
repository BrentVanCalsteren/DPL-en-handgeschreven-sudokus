import torch

from sudokuNeuralNet.NormalNeuralNets import torchNet
from sudokuNeuralNet.NormalNeuralNets.neuralNet import neuralNet


class checkIsCorrect(neuralNet):
    def __init__(self, data_set, lr, epoch):
        super().__init__(epoch,data_set)
        self.model = torchNet.Sudoku_Checker(sudoku_size=self.sqlength**2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCELoss()


    def train_model(self, iter, report):
        for e in range(self.epoch):
            for i in range(iter):
                target, _, input, _ = self.get_random_data_el()
                self.forward(input)
                loss = self.get_loss(target)
                self.update_net(loss)
                if i % report == 0:
                    avg_loss = self.total_loss/report
                    self.avg_loss_list.append(avg_loss)
                    self.report_point_list.append(i+(e*iter))
                    print(f" Iteration: {i},epoch {e}: Avg loss {avg_loss}: "
                          f"\r\n latest output: {self.last_outputValue} \r\n expected: {self.last_targetValue}")
                    self.total_loss = 0



    def forward(self, input):
        sudoku = self.rescale(input)
        torch_input = torch.tensor(sudoku, dtype=torch.float32).view(1, self.sqlength ** 2)
        self.last_outputValue = self.model(torch_input)
        return self.last_outputValue

    def get_loss(self, target):
        self.last_targetValue = torch.tensor([target], dtype=torch.float32).view(1,1)
        loss = self.criterion(self.last_outputValue, self.last_targetValue)
        self.total_loss += loss.item()
        return loss


    def update_net(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()