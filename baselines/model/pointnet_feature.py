"""Adopted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet.py"""
import torch
import torch.nn as nn

class PointNetCls_feature(nn.Module):
    def __init__(self, k=2, use_bn=True, dim=1024):
        super(PointNetCls_feature, self).__init__()
        if dim==1024:
            self.use_bn = use_bn

            if use_bn:
                self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512)
                )
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
            else:
                self.fc1 = nn.Linear(1024, 512)
                self.fc2 = nn.Linear(512, 256)

            self.fc3 = nn.Linear(256, k)
            self.dropout = nn.Dropout(p=0.3)
            self.relu = nn.ReLU(inplace=True)
        if dim==512:
            self.use_bn = use_bn

            if use_bn:
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
            else:
                self.fc2 = nn.Linear(512, 256)

            self.fc3 = nn.Linear(256, k)
            self.dropout = nn.Dropout(p=0.3)
            self.relu = nn.ReLU(inplace=True)
        
        if dim==256:
            self.fc3 = nn.Linear(256, k)

    def forward(self, feature):
        if feature.shape[-1]==1024:
            feature1 = self.relu(self.fc1(feature))
            feature2 = self.dropout(self.fc2(feature1))
            if self.use_bn:
                x = self.bn2(feature2)
            feature3 = self.relu(x)
            x = self.fc3(feature3)
        if feature.shape[-1]==512:
            feature2 = self.dropout(self.fc2(feature))
            if self.use_bn:
                x = self.bn2(feature2)
            feature3 = self.relu(x)
            x = self.fc3(feature3)
        if feature.shape[-1]==256:
            x = self.fc3(feature)
        return x
