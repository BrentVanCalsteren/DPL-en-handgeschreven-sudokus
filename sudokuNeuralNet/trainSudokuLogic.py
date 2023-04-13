import torch
import torch.nn as nn
import dataSets
import torchNet

model = torchNet.Sudoku_Check_Valid(sudoku_size=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.BCELoss()


def train_model(report=100):
    avg_loss = 0.0
    i = 0
    for data in dataSets.sudoku_datasets["train4x4WithEmpty"]:
        avg_loss += updateModel(data).item()
        i+=1
        if i % report == 0:
            print(f"NumberSamples {i}, Avg loss last {report}: {avg_loss/report}")
           # print(torch_target.view(-1))
            #print("Vals:" + str(torch_output.view(-1)))
            avg_loss = 0

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


def main():
    for _ in range(100):
        train_model(1000)
    pass

if __name__ == '__main__':
    main()