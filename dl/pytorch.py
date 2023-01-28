from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

with open("conf.yaml") as fp:
    confg = yaml.safe_load(fp.read())['TRAINING']


def load_data(paths):
    X, y = None, None
    for path in paths:
        data = np.load(path)['arr_0']
        X, y = data[:, 1:], data[:, 0:1]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=confg['SEED'])

    return Xtrain, Xtest, ytrain, ytest


class Net(nn.Module):
    def __init__(self, d_feat: int = 29, d_hid: int = 128) -> None:
        super().__init__()

        self.lin1 = nn.Linear(d_feat, d_hid)
        self.fc1 = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_hid),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(d_hid, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.lin1(x)
        x = self.fc1(x)
        x = self.lin2(x)

        return x


class Data(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device='cuda') -> None:
        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def train(net: nn.Module, dataloader_train: DataLoader, dataset_train: Data):
    net.train()

    loss_train = []
    for data in tqdm(dataloader_train, total=int(len(dataset_train) / confg['BATCH_SIZE'])):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_train.append(loss.item())

    final_loss = sum(loss_train) / len(loss_train)

    return final_loss


def eval(net: nn.Module,  dataloader_test: DataLoader, dataset_test: Data):
    net.eval()

    with torch.no_grad():
        loss_eval = []
        for data in tqdm(dataloader_test, total=int(len(dataset_test) / confg['BATCH_SIZE'])):
            inputs, labels = data

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss_eval.append(loss.item())

        final_loss = sum(loss_eval) / len(loss_eval)

        return final_loss


if __name__ == '__main__':

    root_path = "/media/DataLinux/tcdata/outlet_prediction/res/processed/train"
    paths = Path(root_path).glob("*.npz")

    Xtrain, Xtest, ytrain, ytest = load_data(paths)

    dataset_train, dataset_test = Data(Xtrain, ytrain), Data(Xtest, ytest)
    dataloader_train = DataLoader(dataset_train, batch_size=confg['BATCH_SIZE'], shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=confg['BATCH_SIZE'])

    net = Net()
    net = net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), float(confg['LR']))

    best_loss = 10e10
    for i in range(confg['N_EPOCHS']):
        loss_train = train(net, dataloader_train, dataset_train)

        logger.info(f"Epch {i:2d}: MSE: {loss_train:.5f}")

        if i % 4 == 0:
            loss_eval = eval(net, dataloader_test, dataset_test)

            if loss_eval < best_loss:
                best_loss = loss_eval

                torch.save(net.state_dict(), f"res/best_linear_MSE_{best_loss:.3f}.pth")
