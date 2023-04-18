import torch
import torch.nn as nn
from sudokuNeuralNet.NormalNeuralNets import torchNet, dataSets

model = torchNet.convert_sudoku_image_to_number(sudoku_size=16, image_size=28)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


def train_model(report=100):
    avg_loss = 0.0
    i = 0
    for data in dataSets.sudoku_datasets["train4x4EmptyTrue"]:
        avg_loss += updateModel(data).item()
        i+=1
        if i % report == 0:
            print(f"NumberSamples {i}, Avg loss last {report}: {avg_loss/report}")
           # print(torch_target.view(-1))
            #print("Vals:" + str(torch_output.view(-1)))
            avg_loss = 0

def updateModel(data):
    sudokuInfo = list(list(data.values())[0])
    sudokuImage = sudokuInfo[0]
    sudokuUnsolved = sudokuInfo[1]
    torch_input = convert2image(sudokuImage)
    torch_target = convert2ValueMatrix(rescale(sudokuUnsolved))
    torch_output = model(torch_input)
    loss = criterion(torch_output, torch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def rescale(sudoku):
    l = len(sudoku)
    return [(x/l) for row in sudoku for x in row]


def convert2ValueMatrix(sudokuUnsolved):
    matrix2d = list()
    ##flat_list = [item for sublist in sudokuUnsolved for item in sublist]
    for e in sudokuUnsolved:
        zeroList = [0] * 10
        zeroList[int(e*4)] = 1
        matrix2d.append(zeroList)
    return torch.tensor(matrix2d, dtype=torch.float32).view(1,1,16,10)




def main():
    #for _ in range(100):
    for _ in range(5):
        train_model(1000)
    #pass
    torchNet.save_model(model, "test")

if __name__ == '__main__':
    main()
