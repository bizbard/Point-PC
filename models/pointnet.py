import torch
import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    def __init__(self, feature_channel=256, num_class=55):
        super().__init__()
        self.feature_channel = feature_channel
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, fea):
        x = F.relu(self.bn1(self.fc1(fea)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, encoder_channel=256):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B N 3
            -----------------
            feature_global : B C
        '''
        b, n, _ = point_groups.shape
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 256
        return feature_global

