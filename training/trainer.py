from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import xgboost as xgb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from .ptlight import LitData, LitModel


def get_training_dat(path_train: str, path_val: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_train = np.load(path_train)['arr_0']
    Xtrain, ytrain = df_train[:, 1:], df_train[:, 0]

    df_test = np.load(path_val)['arr_0']
    Xtest, ytest = df_test[:, 1:], df_test[:, 0]

    return Xtrain, ytrain, Xtest, ytest


def linear_regression(Xtrain, ytrain, Xtest, ytest, **kwargs):
    regression = LinearRegression(positive=True, n_jobs=10)
    regression.fit(Xtrain, ytrain)

    pred = regression.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")


def random_forest(Xtrain, ytrain, Xtest, ytest, **kwargs):
    regression = RandomForestRegressor(n_estimators=100, random_state=37, n_jobs=10)
    regression.fit(Xtrain, ytrain)

    pred = regression.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")


def xgboost(Xtrain, ytrain, Xtest, ytest, **kwargs):
    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50)
    reg.fit(Xtrain, ytrain,
            eval_set=[(Xtrain, ytrain), (Xtest, ytest)],
            verbose=False)

    pred = reg.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")


def dl_SimpleMLP(Xtrain, ytrain, Xtest, ytest, conf_training, **kwargs):
    # Define modules for training
    litmodel = LitModel(conf_training)
    lidata = LitData(conf_training, Xtrain, ytrain, Xtest, ytest)

    # Define callbacks
    now = datetime.now()
    dt_str = now.strftime("%m%d_%H%M%S")

    path_ckpt = Path(conf_training['CKPT']) / dt_str
    callback_modelckpt = ModelCheckpoint(str(path_ckpt), monitor="msle_val", save_top_k=1,
                                         mode='min', filename="{epoch}-{msle_val:.6f}")
    callback_lr_monitor = LearningRateMonitor(logging_interval='step')

    logger_tboard = TensorBoardLogger(conf_training['LOGGER'], default_hp_metric=True, version=dt_str)

    trainer = pl.Trainer(
        devices=1,
        gradient_clip_val=1,
        accelerator="gpu",
        check_val_every_n_epoch=5,
        default_root_dir="ckpt/",
        max_epochs=conf_training['N_EPOCHS'],
        log_every_n_steps=1,
        callbacks=[callback_modelckpt, callback_lr_monitor],
        logger=logger_tboard
    )
    trainer.fit(litmodel, datamodule=lidata)
