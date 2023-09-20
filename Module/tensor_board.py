# -*- coding = utf-8 -*-
# @Author : Jingbo Su
# @File : tensor_board.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

import time
from tqdm import tqdm

# Hypermeters
input_size = 28
seq_len = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.01
# learning_rate = [0.1, 0.01, 0.001, 0.0001]
batch_size = 16
# batch_size = [8, 16, 32, 64, 128]
num_epochs = 3

# Global parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_file = 'runs/checkpoint.pth'
load_model = False


class Net(nn.Module):
    def __init__(self, input_size: int, sequence_len: int, hidden_size: int, num_layers: int, num_classes: int, rnn_model: str):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_len, num_classes)
        self.model = rnn_model

    def forward(self, x):
        h_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)   # (D(bi=2)*num_layers, batch_size, h_out)
        c_0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)   # (D(bi=2)*num_layers, batch_size, h_cell)
        if self.model == 'rnn':
            out, _ = self.rnn(x, h_0)
        elif self.model == 'gru':
            out, _ = self.gru(x, h_0)
        else:
            out, _ = self.lstm(x, (h_0, c_0))   # (batch_size, num_layers, D * h_out)
        out = out.reshape(out.size(0), -1)  # (batch_size, other_dim)
        out = self.fc(out)
        return out


# DataLoader
train_data = datasets.MNIST(
    root='./dataset',
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(
    root='./dataset',
    train=False,
    transform=ToTensor()
)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


def check_accuracy(loader, model, device):
    if loader.dataset.train:
        print('Checking training set accuracy...')
    else:
        print('Checking test set accuracy...')

    correct = 0.
    total = 0.
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)
            result = model(x)
            _, y_pred = result.max(1)
            correct += (y_pred == y).sum().item()
            total += y_pred.size(0)

    model.train()   # Toggle model back to train
    return correct / total

def save_checkpoint(state: dict, filename=save_file) -> None:
    print('Saving checkpoint...')
    torch.save(state, filename)

def load_checkpoint(checkpoint) -> None:
    print('Loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


step = 0

if __name__ == '__main__':
    print(f'device: {device}')
    rnn_models = ['rnn', 'gru', 'lstm']
    for i in range(3):
        print('='*32)
        print(f'model: {rnn_models[i]}')

        # Model initialization
        model = Net(input_size=input_size, sequence_len=seq_len, hidden_size=hidden_size,
                    num_layers=num_layers, num_classes=num_classes, rnn_model=rnn_models[i])
        model.to(device)
        model.train()

        # Loss and Optimizer
        losses = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Default: [betas=(0.9, 0.999), eps=1e-8, lr=0.001, weight_decay=0.0]
        writer = SummaryWriter(f'runs/MNIST/{rnn_models[i]}')

        # Training
        if load_model:
            torch.load(save_file)
        start = time.time()
        for epoch in range(num_epochs):
            if epoch != 0 and epoch % num_epochs == 0:
                state_dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_checkpoint(state_dict)

            for _, (x, y) in enumerate(tqdm(train_loader)):
                x = x.to(device).squeeze(1)
                y = y.to(device)

                # Forward
                result = model(x)   # (batch_size, 10)
                loss = criterion(result, y)
                _, y_pred = result.max(1)
                correct = (y_pred == y).sum()
                acc = float(correct) / float(x.size(0))
                losses.append(loss.item())

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Training loss', loss.item(), global_step=step)
                writer.add_scalar('Training accuracy', acc, global_step=step)
                writer.add_histogram('fc', model.fc.weight)
                step += 1

        end = time.time()
        delta = end - start

        train_acc = check_accuracy(train_loader, model, device)
        test_acc = check_accuracy(test_loader, model, device)

        writer.add_hparams({'Training model': rnn_models[i]}, {'Accuracy': train_acc, 'Loss': sum(losses)/len(losses)})

        print(f'Time Cost = {delta:.2f}s')
        print(f'Train Acc = {train_acc*100.:.2f}')
        print(f'Test Acc = {test_acc*100.:.2f}')