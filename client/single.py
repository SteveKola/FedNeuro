from os import environ
from LSUV import LSUVinit

import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


@model_wrapper 
class ModelSpace(nn.Module):
    def __init__(self):
        super(ModelSpace, self).__init__()
        self.conv1 = nn.LayerChoice([
            nn.Conv2d(1, 32, 3, 1),
            nn.Conv2d(1, 32, 5, 3)
        ])  # try 3x3 kernel and 5x5 kernel
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

# I need to give the model MNIST data to init weights. This will make search take less time.
# I also need to
# print(model_space)
# to see if everything works up to this point, but I hate working with Python locally. 
# I think I need to move to some cloud environment because installing packages locally sucks.
model = LSUVinit(model_space,data)




# dataset_train = MNIST(root="./data", train=True, download=True, transform=train_transform)
# dataset_valid = MNIST(root="./data", train=False, download=True, transform=valid_transform)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)


# # use NAS here
# def top1_accuracy(output, target):
#     # this is the function that computes the reward, as required by ENAS algorithm
#     batch_size = target.size(0)
#     _, predicted = torch.max(output.data, 1)
#     return (predicted == target).sum().item() / batch_size

# def metrics_fn(output, target):
#     # metrics function receives output and target and computes a dict of metrics
#     return {"acc1": top1_accuracy(output, target)}

# from nni.algorithms.nas.pytorch import enas
# trainer = enas.EnasTrainer(model,
#                            loss=criterion,
#                            metrics=metrics_fn,
#                            reward_function=top1_accuracy,
#                            optimizer=optimizer,
#                            batch_size=128
#                            num_epochs=10,  # 10 epochs
#                            dataset_train=dataset_train,
#                            dataset_valid=dataset_valid,
#                            log_frequency=10)  # print log every 10 steps
# trainer.train()  # training
# trainer.export(file="model_dir/final_architecture.json")  # export the final architecture to file
