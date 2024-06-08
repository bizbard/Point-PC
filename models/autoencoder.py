import sys
import torch
import torch.nn as nn
from models.pct import PCTransformer
from models.foldnet import Fold


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = PCTransformer(config=self.config.value_encoder)
        self.decoder = Fold(config=self.config.decoder)

    def forward(self, inp):
        coarse_point_cloud, feature, _ = self.encoder(inp)
        B, M, _ = coarse_point_cloud.shape
        xyz = self.decoder(feature.reshape(B*M, -1)).reshape(B, M, -1, 3)
        fine = (xyz + coarse_point_cloud.unsqueeze(2)).reshape(B, -1, 3)
        return coarse_point_cloud, fine

