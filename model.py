#%%
import torch
from torch import nn



class ResNet(nn.Module):
    def __init__(self):
        super().__init__(self)
        l1 = nn.Sequential(
            nn.Conv2d(192, 48),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )



    def forward(self, X):
        X = self.l1(X)