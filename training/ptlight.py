
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredLogError


class Data(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LitData(pl.LightningDataModule):
    def __init__(self, confg, Xtrain, ytrain, Xtest, ytest) -> None:
        super().__init__()

        self._confg = confg

        self.data_train: Union[Data, None]
        self.data_val: Union[Data, None]
        self.data_train = Data(Xtrain, ytrain)
        self.data_val = Data(Xtest, ytest)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self._confg['BATCH_SIZE'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self._confg['BATCH_SIZE'], num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_val, batch_size=self._confg['BATCH_SIZE'], num_workers=4)


class Net(nn.Module):
    def __init__(self, d_feat: int = 32, d_hid: int = 128) -> None:
        super().__init__()

        self.lin1 = nn.Linear(d_feat, d_hid)
        self.fc1 = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.LeakyReLU(),
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, d_hid),
            nn.LeakyReLU(),
            nn.LayerNorm(d_hid)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(d_hid, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.lin1(x)
        x = self.fc1(x)
        x = self.lin2(x)

        x = x.squeeze(-1)

        return x


class LitModel(pl.LightningModule):
    def __init__(self, confg) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._confg = confg

        self.net = Net(confg['D_FEAT'])
        self.criterion = nn.MSELoss()
        self.msle = MeanSquaredLogError()

    def forward(self, batch):
        inputs, labels = batch

        outputs = self.net(inputs)

        return outputs, labels

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        self.log("loss_train", loss, prog_bar=False,
                 on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_epoch_end(self, outputs: list) -> None:
        pred = torch.cat([x[0] for x in outputs])
        gt = torch.cat([x[1] for x in outputs])

        metric = self.msle(torch.exp(pred), torch.exp(gt))

        self.log("msle_val", metric)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), float(self._confg['LR']), weight_decay=5e-4)
        # optimizer = optim.SGD(self.parameters(), float(self._confg['LR']))

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1)

        return [optimizer], [lr_scheduler]
