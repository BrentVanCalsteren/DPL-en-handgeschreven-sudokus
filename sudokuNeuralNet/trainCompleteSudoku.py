import torch
import torch.nn as nn
import dataSets
import torchNet


model = torchNet.Complete_Sudoku(sudoku_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
last_targetValue = None
last_outputValue = None
number = 0
criterion = nn.MSELoss()


def main():
    #for _ in range(100):
    for _ in range(20):
        train_model(1000)
    #pass
    torchNet.save_model(model, "test")

def train_model(report=100, datasetName="train4x4EmptyTrue"):
    global number, last_outputValue, last_targetValue
    loss = list()
    avg_loss = 0.0
    for data in dataSets.sudoku_datasets[datasetName]:
        avg_loss += updateModel(data).item()
        number += 1
        if number % report == 0:
            print(f"Number samples: {number}, Avg loss last {report}: {avg_loss/report}")
            print("Target:" + str(last_targetValue))
            print("Result:" + str(last_outputValue))
            avg_loss = 0

def updateModel(data):
    global last_outputValue, last_targetValue
    label = list(data.keys())[0]
    p = float(label == 'True')
    data = list(list(data.values())[0])
    sudokuUnsolved = data[1]
    l = len(sudokuUnsolved)
    sudokuSolved = data[2]
    sudokuUnsolved = rescale(sudokuUnsolved)
    sudokuSolved = rescale(sudokuSolved)
    torch_input = torch.tensor(sudokuUnsolved, dtype=torch.float32).view(1,1,l,l)
    torch_target = torch.tensor(sudokuSolved, dtype=torch.float32).view(1,l**2)
    torch_output = model(torch_input)
    last_outputValue = torch_output.tolist()
    last_targetValue = torch_target.tolist()
    loss = criterion(torch_output, torch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def rescale(sudoku):
    l = len(sudoku)
    return [(x/l) for row in sudoku for x in row]


if __name__ == '__main__':
    main()