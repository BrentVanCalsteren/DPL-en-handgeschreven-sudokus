import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SudokuSolver(nn.Module):
    def __init__(self):
        super(SudokuSolver, self).__init__()
        self.fc1 = nn.Linear(81, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 81)

    def forward(self, x):
        x = x.view(-1, 81)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, train_loader, num_epochs):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, loss: {running_loss / (i + 1)}")

def main():
    model = SudokuSolver()
    optimizer = optim.Adam(model.parameters())
    train_model(model, optimizer, 1)

if __name__ == '__main__':
    main()