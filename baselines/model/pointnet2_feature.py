"""Adopted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


class PointNet2ClsSsg_feature(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet2ClsSsg_feature, self).__init__()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, feature):
        x = self.fc3(feature)
        return x
