import torch.nn as nn

def create_model():
    return nn.Sequential(
        nn.Linear(3,64),
        nn.Tanh(),
        nn.Linear(64,100), 
        nn.Tanh(),
        nn.Linear(100,64), 
        nn.Tanh(),
        nn.Linear(64,1)
    )