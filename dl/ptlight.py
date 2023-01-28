from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset


class Data(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LitData(pl.LightningDataModule):
    def __init__(self, confg, paths) -> None:
        super().__init__()

        self._confg = confg
        self._paths = paths

        self.data_train: Union[Data, None]
        self.data_val: Union[Data, None]
        self.data_train, self.data_val = None, None

    def setup(self, stage) -> None:
        X, y = None, None
        for path in self._paths:
            data = np.load(path)['arr_0']
            X, y = data[:, 1:], data[:, 0:1]

        Xtrain, Xval, ytrain, yval = train_test_split(
            X, y, test_size=0.2, random_state=self._confg['SEED'])

        self.data_train = Data(Xtrain, ytrain)
        self.data_val = Data(Xval, yval)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self._confg['BATCH_SIZE'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self._confg['BATCH_SIZE'], num_workers=4)


class Net(nn.Module):
    def __init__(self, d_feat: int = 28, d_hid: int = 128) -> None:
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


class LitModel(pl.LightningModule):
    def __init__(self, confg) -> None:
        super().__init__()

        self._confg = confg

        self.net = Net()
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        self.log("loss_train", loss, prog_bar=True,
                 on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        self.log('loss_val', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), float(self._confg['LR']))

        return optimizer


if __name__ == '__main__':
    # Read config
    with open("conf.yaml") as fp:
        confg = yaml.safe_load(fp.read())['TRAINING']

    root_path = "/media/DataLinux/tcdata/outlet_prediction/res/processed/train"
    paths = Path(root_path).glob("*.npz")

    # Define modules for training
    litmodel = LitModel(confg)
    lidata = LitData(confg, paths)

    callback_modelckpt = ModelCheckpoint(confg['CKPT'], monitor="loss_val", save_top_k=1,
                                         mode='min', filename="{epoch}-{val_loss:.6f}")
    now = datetime.now()
    dt_str = now.strftime("%m%d_%H%M%S")
    logger_tboard = TensorBoardLogger(confg['LOGGER'], default_hp_metric=False, version=dt_str)

    trainer = pl.Trainer(
        devices=1,
        gradient_clip_val=0.5,
        accelerator="gpu",
        check_val_every_n_epoch=5,
        default_root_dir="ckpt/",
        max_epochs=confg['N_EPOCHS'],
        log_every_n_steps=1,
        callbacks=callback_modelckpt,
        logger=logger_tboard
    )
    trainer.fit(litmodel, datamodule=lidata)
