import torch
import torch.nn as nn
import dataSets
import torchNet

model = torchNet.convert_sudoku_image_to_number(sudoku_size=16, image_size=28)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
    sudokuInfo = list(list(data.values())[0])
    sudokuImage = sudokuInfo[0]
    sudokuUnsolved = sudokuInfo[1]
    torch_input = convert2image(sudokuImage)
    torch_target = convert2ValueMatrix(sudokuUnsolved)
    torch_output = model(torch_input)
    loss = criterion(torch_output, torch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def convert2ValueMatrix(sudokuUnsolved):
    matrix2d = list()
    flat_list = [item for sublist in sudokuUnsolved for item in sublist]
    for e in flat_list:
        zeroList = [0] * 10
        zeroList[e] = 1
        matrix2d.append(zeroList)
    return torch.tensor(matrix2d, dtype=torch.float32).view(1,1,16,10)

def convert2image(sudoku):
    list = [load_image(x) for row in sudoku for x in row]
    return torch.stack(list, dim=1)
    #tuples = [torch.squeeze(torch.tensor(el)).tolist() for el in list]
    #return tuples

def load_image(index):
    if index == 'empty':
        return torch.zeros((1, 28, 28))
    else:
        image, label = dataSets.mnist["train"][index]
        return image


def main():
    for _ in range(100):
        train_model(1000)
    pass

if __name__ == '__main__':
    main()
