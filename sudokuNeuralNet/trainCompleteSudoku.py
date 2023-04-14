import random

import torch
import torch.nn as nn
import dataSets
import torchNet
from sudokuNeuralNet import plotter

model = None
optimizer = None
last_targetValue = None
last_outputValue = None
number = 0
criterion = None


def init(sudoku_result=False):
    global model, optimizer, criterion
    if sudoku_result:
        model = torchNet.Sudoku_Solver(sudoku_size=81)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
    else:
        model = torchNet.Sudoku_Checker(sudoku_size=81)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCELoss()

def main():
    init(sudoku_result=False)
    losses = []
    iterations = []
    for _ in range(5):
        loss, iteration = train_model(report=1000, datasetName="train9x9NoEmpty",sudoku_result=False)
        losses += loss
        iterations += iteration
    plotter.plot( numbers1=iterations,numbers2=losses, label1="iterations", label2="losses",)

def train_model(report=100, datasetName="train4x4EmptyTrue",sudoku_result=False):
    global number, last_outputValue, last_targetValue
    loss = list()
    iteration = list()
    avg_loss = 0.0
    set = dataSets.sudoku_datasets[datasetName]
    l = len(set)
    for _ in set:
        data = random.choice(set)
        avg_loss += updateModel(data, sudoku_result).item()
        number += 1
        if number % report == 0:
            print(f"Number samples: {number}, Avg loss last {report}: {avg_loss/report}")
            print("Target:" + str(last_targetValue))
            print("Result:" + str(last_outputValue))
            loss.append(avg_loss/report)
            iteration.append(number)
            avg_loss = 0
    return loss, iteration

def updateModel(data, sudoku_result=False):
    global last_outputValue, last_targetValue
    label = list(data.keys())[0]
    p = float(label == 'True')
    data = list(list(data.values())[0])
    sudokuUnsolved = data[1]
    l = len(sudokuUnsolved)
    sudokuSolved = data[2]
    sudokuUnsolved = rescale(sudokuUnsolved)
    sudokuSolved = rescale(sudokuSolved)
    torch_input = torch.tensor(sudokuUnsolved, dtype=torch.float32).view(1, l**2)
    if sudoku_result:
        torch_input = torch.tensor(sudokuUnsolved, dtype=torch.float32).view(1, 1, l, l)
    torch_target_sudoku = torch.tensor(sudokuSolved, dtype=torch.float32).view(1,l**2)
    torch_target_valid = torch.tensor([p], dtype=torch.float32).view(1,1)
    torch_output = model(torch_input)
    last_outputValue = torch_output.tolist()
    if sudoku_result:
        last_targetValue = torch_target_sudoku.tolist()
        loss = criterion(torch_output, torch_target_sudoku)
    else:
        last_targetValue = torch_target_valid.tolist()
        loss = criterion(torch_output, torch_target_valid)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def rescale(sudoku):
    l = len(sudoku)
    return [(x/l) for row in sudoku for x in row]


if __name__ == '__main__':
    main()