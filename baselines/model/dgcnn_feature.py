"""Adopted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNN_feature(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNN_feature, self).__init__()
        
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, feature1):
        x = self.linear3(feature1)
        return x
