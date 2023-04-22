import torch
from sudokuNeuralNet.NormalNeuralNets import torchNet
from sudokuNeuralNet.NormalNeuralNets.neuralNet import neuralNet


class mnistNet(neuralNet):
    def __init__(self, data_set, lr, epoch):
        super().__init__(epoch, data_set)
        self.model = torchNet.MNIST_Net(n=self.sqlength)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()


    def train_model(self, iter, report):
        for e in range(self.epoch):
            i = 0
            for _ in range(iter):
                _, input, target, _ = self.get_random_data_el()
                for k in range(self.sqlength):
                    for l in range(self.sqlength):
                        i+=1
                        inp = input[k][l]
                        self.forward(inp)
                        loss = self.get_loss(target[k][l])
                        self.update_net(loss)
                        if i % report == 0:
                            avg_loss = self.total_loss/report
                            self.avg_loss_list.append(avg_loss)
                            self.report_point_list.append(i+(e*iter))
                            print(f" Iteration: {i},epoch {e}: Avg loss {avg_loss}: "
                                  f"\r\n latest output: {self.last_outputValue} \r\n expected: {self.last_targetValue}"
                                  f"\r\n overal ACC: {self.get_accuracy()}")
                            self.total_loss = 0



    def forward(self, input):
        number = input.view(1,28,28)
        self.last_outputValue = self.model(number)
        return self.last_outputValue

    def get_loss(self, target):
        target = self.convertValue2List(target,n=self.sqlength)
        self.last_targetValue = torch.tensor(target, dtype=torch.float32).view(1,self.sqlength)
        loss = self.criterion(self.last_outputValue, self.last_targetValue)
        self.total_loss += loss.item()
        return loss


    def update_net(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def convertValue2List(self,value,n=10):
        zeroList = [0] * n
        zeroList[int(value-1)] = 1.0
        return zeroList


    def convertList2Value(self,list):
        index = 0
        maxvalue = 0.0
        if len(list) == 1:
            list = list[0]
        for i in range(len(list)):
            val = list[i].item()
            if val > maxvalue:
                index = i+1
                maxvalue = val
        return index

    def get_accuracy(self, name ="test100-16-4x4"):
        dset = self.open_dataset(name)
        total = 0
        correct = 0
        for ddata in dset:
            _, input, target, _ = self.convert_data(ddata)
            for k in range(self.sqlength):
                for l in range(self.sqlength):
                    inp = input[k][l]
                    number = inp.view(1, 28, 28)
                    output = self.model(number)
                    out = self.convertList2Value(output.detach().numpy())
                    comp = target[k][l]
                    total += 1
                    if int(out) == int(comp):
                        correct += 1
        return correct/total