import numpy as np
import torch.nn as nn

def train(n_epochs, optimizer, model, loss_fn, Xtrain, Xval, ytrain, yval):
    for epoch in range(1, n_epochs + 1):

        out = model(X_train)
        loss = loss_fn(out, ytrain)

        

        
if __name__ == '__main__':

    X = [x for x in range(11)]
    y = [1.6*x + 4 + np.random.normal(10, 1) for x in X]

    X_train = X[:9]
    y_train = y[:9]
    X_val = X[9:]
    y_val = X[9:]

    seq_model = nn.Sequential(
        nn.Linear(1, 13),
        nn.Tanh(),
        nn.Linear(13, 1)
        )

    seq_model.train()



