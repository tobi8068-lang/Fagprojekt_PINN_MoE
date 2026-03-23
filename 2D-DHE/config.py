import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_GRID = 50
N_PLOT = 100

ALPHA = 8.4e-5
LR = 1e-3
EPOCHS = 5001