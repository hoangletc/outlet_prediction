import argparse

import numpy as np
import yaml
from loguru import logger

from training import trainer

MODELS = {
    'lin_reg': "linear_regression",
    'forest': "random_forest",
    'xg': "xgboost",
    'mlp': "dl_SimpleMLP"
}


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training trigger')

    parser.add_argument('--train-data', type=str, help='Path to training data')
    parser.add_argument('--val-data', type=str, help='Path to training data')
    parser.add_argument('--model', '-m', type=str, help='Path to training data')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = init_args()

    path = "/media/DataLinux/tcdata/outlet_prediction/conf.yaml"
    with open(path) as fp:
        config = yaml.safe_load(fp)

    logger.info("Load training data")
    # Xtrain, ytrain, Xtest, ytest = trainer.get_training_dat(args.train_data, args.val_data)
    Xtrain = np.load("Xtrain.npz")['arr_0'].astype('float32')
    ytrain = np.load("ytrain.npz")['arr_0'].astype('float32')
    Xtest = np.load("Xval.npz")['arr_0'].astype('float32')
    ytest = np.load("yval.npz")['arr_0'].astype('float32')

    logger.info("Start training")
    # trainer.linear_regression(Xtrain, ytrain, Xtest, ytest)
    # trainer.random_forest(Xtrain, ytrain, Xtest, ytest)
    # trainer.xgboost(Xtrain, ytrain, Xtest, ytest)
    # trainer.dl_SimpleMLP(conf_training, Xtrain, ytrain, Xtest, ytest)

    method = MODELS[args.model]
    getattr(trainer, method)(Xtrain, ytrain, Xtest, ytest, conf_training=config['TRAINING'])

# python trigger_train.py -m mlp \
#     --train-data res/processed/v1.1/train.npz \
#     --val-data res/processed/v1.1/test.npz
