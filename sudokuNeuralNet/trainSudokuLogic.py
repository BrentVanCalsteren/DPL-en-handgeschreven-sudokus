import torch
import torch.nn as nn
import dataSets


class SudokuSolver(nn.Module):
    def __init__(self):
        super(SudokuSolver, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = 10 * self.sigmoid(x) - 3
        return x


def train_model(model,report=100, epochs=1):
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


def main():
    model = SudokuSolver()
    train_model(model,1000, 100)

if __name__ == '__main__':
    main()