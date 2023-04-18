import torch
import torch.nn as nn
from sudokuNeuralNet.NormalNeuralNets import torchNet, dataSets
from sudokuNeuralNet import plotter

model = torchNet.Sudoku_Check_Valid(sudoku_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.BCELoss()

def main():
    losses = []
    iterations = []
    loss, iteration = train_model(report=1000, datasetName="train4x4Empty")
    losses += loss
    iterations += iteration
    plotter.plot( numbers1=iterations,numbers2=losses, label1="iterations", label2="losses",)
    #result = test(model)

def train_model(report=100, datasetName="train4x4Empty"):
    loss = list()
    iteration = list()
    avg_loss = 0.0
    i = 0
    for data in dataSets.sudoku_datasets[datasetName]:
        los = updateModel(data).item()
        avg_loss += los
        i+=1
        if i % report == 0:
            print(f"NumberSamples {i}, Avg loss last {report}: {avg_loss/report}")
            loss.append(avg_loss/report)
            iteration.append(i)
           # print(torch_target.view(-1))
            #print("Vals:" + str(torch_output.view(-1)))
            avg_loss = 0
    return loss, iteration

def updateModel(data):
    label = list(data.keys())[0]
    p = float(label == 'True')
    sudokuUnsolved = list(list(data.values())[0])[1]
    torch_input = torch.tensor(sudokuUnsolved, dtype=torch.float32).view(-1)
    torch_target = torch.tensor([p], dtype=torch.float32)
    torch_output = model(torch_input)
    loss = criterion(torch_output, torch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss




if __name__ == '__main__':
    main()