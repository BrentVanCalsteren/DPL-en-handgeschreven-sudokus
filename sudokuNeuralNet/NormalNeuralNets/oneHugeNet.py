import torch

from sudokuNeuralNet.NormalNeuralNets import torchNet
from sudokuNeuralNet.NormalNeuralNets.neuralNet import neuralNet


class oneHugeNet(neuralNet):
    def __init__(self, data_set, lr, epoch):
        super().__init__(epoch,data_set)
        self.model = torchNet.oneHugeNet2(self.sqlength ** 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCELoss()


    def train_model(self, iter, report,accu=0.99):
        for e in range(self.epoch):
            for i in range(1,iter+1):
                target, input, _, _ = self.get_random_data_el()
                self.forward(input)
                loss = self.get_loss(target)
                self.update_net(loss)
                if i % report == 0:
                    avg_loss = self.total_loss/report
                    self.avg_loss_list.append(avg_loss)
                    self.report_point_list.append(i+(e*iter))
                    print(f" Iteration: {i},epoch {e}: Avg loss {avg_loss}: "
                          f"\r\n latest output: {self.last_outputValue} \r\n expected: {self.last_targetValue}"
                          f"\r\n overal ACC: {self.get_accuracy()}")
                    if (self.accu[-1] > accu):
                        return
                    self.total_loss = 0



    def forward(self, input):
        #torch_input = torch.tensor(input, dtype=torch.float32).view(1, self.sqlength ** 2,28,28)
        self.last_outputValue = self.model(input.view(self.sqlength ** 2,28,28))
        return self.last_outputValue

    def get_loss(self, target):
        self.last_targetValue = torch.tensor([target], dtype=torch.float32)
        loss = self.criterion(self.last_outputValue, self.last_targetValue)
        self.total_loss += loss.item()
        return loss


    def update_net(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_accuracy(self, name ="test100-16-4x4"):
        dset = self.open_dataset(name)
        total = 0
        correct = 0
        for ddata in dset:
            target, input, _, _ = self.convert_data(ddata)
            output = self.model(input.view(self.sqlength ** 2,28,28)).item()
            comp = target
            total += 1
            if round(output) == int(comp):
                correct += 1
        accu = correct/total
        self.accu.append(accu)
        return accu
