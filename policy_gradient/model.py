import math

import torch
import torch.nn as nn
from torch.nn import functional as F

def init_weights(module):
    for pn, p in module.named_parameters():
        if(pn.endswith("weight")):
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        if(pn.endswith("bias")):
            torch.nn.init.zeros_(p)
    return

class smolMLP(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        # indexing self.layers allows using only those layers
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_out, bias=True),
            nn.Softmax(dim=-1)
        )
        init_weights(self)

    def forward(self, x):
        return self.layers(x)

class prob_MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        # indexing self.layers allows using only those layers
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_in * 4, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_in * 4, d_out, bias=True),
            nn.Softmax(dim=-1)
        )
        init_weights(self)

    def forward(self, x):
        return self.layers(x)
    
class MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        # indexing self.layers allows using only those layers
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_in * 4, bias=True),
            nn.ReLU(),
            # Dropout in MLPs is usually applied after an activation function
            # sometimes its applied to the input as a form of data noising
            # applying it to a final output is usually a bad idea
            # in Transformers the output of the mlp is not final, which is why its done there
            nn.Dropout(dropout),
            nn.Linear(d_in * 4, d_out, bias=True),
        )
        init_weights(self)

    def forward(self, x):
        return self.layers(x)
