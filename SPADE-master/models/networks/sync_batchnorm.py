import torch
import torch.nn as nn

class SynchronizedBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        return super().forward(x)
