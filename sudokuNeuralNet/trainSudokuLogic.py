import numpy
import torch
import torch.nn as nn
import dataSets


def train_model_fill_sudoku(model,report=100, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    length = dataSets.getSudokuSetLenght("train")
    for epoch in range(epochs):
        avg_loss = 0.0
        j = 0
        for i in range(length):
            j+=1
            dic = dataSets.getSudokuOnIndexData("train", i)
            label = list(dic.keys())[0]
            if label == 'True':
                sudokuAndSolved = list(list(dic.values())[0])
                sudoku = sudokuAndSolved[1]
                torch_input = torch.tensor(sudoku, dtype=torch.float32)
                target = sudokuAndSolved[2]
                torch_target = torch.tensor(target, dtype=torch.float32)
                torch_output = model(torch_input)
                loss = criterion(torch_output.view(-1), torch_target.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                if j > report:
                    print(f"NumberSamples {i}, Avg loss last {report}: {avg_loss/report}")
                    print(torch_target.view(-1))
                    print("Rounded vals:" + str(torch.round(torch_output).view(-1)))
                    j = 0
                    avg_loss = 0

def train_model_check_sudoku(model,report=100, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    length = dataSets.getSudokuSetLenght("train")
    for epoch in range(epochs):
        avg_loss = 0.0
        j = 0
        for i in range(length):
            j+=1
            dic = dataSets.getSudokuOnIndexData("train", i)
            label = list(dic.keys())[0]
            p = float(label == 'True')
            sudokuAndSolved = list(list(dic.values())[0])
            sudoku = sudokuAndSolved[1]
            torch_input = torch.tensor(sudoku, dtype=torch.float32)
            target = [p]
            torch_target = torch.tensor(target, dtype=torch.float32)
            torch_output = model(torch_input)
            loss = criterion(torch_output.view(-1), torch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if j > report:
                print(f"NumberSamples {i}, Avg loss last {report}: {avg_loss/report}")
                print(torch_target.view(-1))
                print("Vals:" + str(torch_output.view(-1)))
                j = 0
                avg_loss = 0

def main():
    model = SudokuChecker()
    train_model_check_sudoku(model,1000, 10)
    pass

if __name__ == '__main__':
    main()