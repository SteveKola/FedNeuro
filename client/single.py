from os import environ
from re import search
from LSUV import LSUVinit

import nni
import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
        super(ModelSpace, self).__init__()
        self.conv1 = nn.LayerChoice(
            [nn.Conv2d(1, 32, 3, 1), nn.Conv2d(1, 32, 5, 3)]
        )  # try 3x3 kernel and 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        self.skipcon = nn.InputChoice(n_candidates=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x0 = self.skipcon([x])  # choose one or none from [x]
        x = self.conv3(x)
        if x0 is not None:  # skipconnection is open
            x += x0
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model_space = ModelSpace()

# we need to give the model MNIST data to initialize weights. This will make search take less time.
training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
model = LSUVinit(model_space, training_data)

search_strategy = strategy.RegularizedEvolution(
    optimize_mode="maximize", population_size="300", cycles="100", on_failure="worst"
)

import nni.retiarii.evaluator.pytorch.lightning as pl
from torchvision import transforms

transform = nni.trace(
    transforms.Compose,
    [
        nni.trace(transforms.ToTensor()),
        nni.trace(transforms.Normalize, (0.1307,), (0.3081,)),
    ],
)
train_dataset = nni.trace(
    model, root="data/mnist", train=True, download=True, transform=transform
)
test_dataset = nni.trace(
    model, root="data/mnist", train=False, download=True, transform=transform
)

# pl.DataLoader and pl.Classification is already traced and supports serialization.
evaluator = pl.Classification(
    train_dataloaders=pl.DataLoader(train_dataset, batch_size=100),
    val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
    max_epochs=10,
)

# using https://nni.readthedocs.io/en/stable/nas/evaluator.html
