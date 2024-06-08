import sys

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from models.pct import PCTransformer
from models.foldnet import Fold
from knn_cuda import KNN
knn = KNN(k=1, transpose_mode=False)


class MemPtc(nn.Module):

    def __init__(self, args, config, pre_ve=False):
        super().__init__()

        self.args = args
        self.config = config
        self.value_encoder = PCTransformer(config=self.config.value_encoder)
        if pre_ve:
            self.load_vepretrain(args)

        self.decoder = Fold(config=self.config.decoder)

        self.decrease_fea_1 = nn.Sequential(
            nn.Conv1d(384*3, 384, 1),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(384, 384, 1)
        )
        self.reduce_fea_1 = nn.Linear(384*2, 384)

        self.decrease_ctr_1 = nn.Sequential(
            nn.Conv1d(3 * 3, 3, 1),
            nn.BatchNorm1d(3),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(3, 3, 1)
        )
        self.reduce_ctr_1 = nn.Linear(3 * 2, 3)

        self.downsample1 = torch.nn.Sequential(
            torch.nn.Conv1d(128 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.downsample2 = torch.nn.Sequential(
            torch.nn.Conv1d(128+128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def load_vepretrain(self, args):
        ckpt = torch.load(args.ve_ckpt)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['ve_encoder'].items()}
        self.value_encoder.load_state_dict(base_ckpt, strict=True)
        # for name, param in self.value_encoder.named_parameters():
        #     param.requires_grad = False

    def square_distance(self, src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def get_farest_feature(self, coor_q, coor_k, x_k):
        """
            coor_q:  (B, 32, 3).
            coor_k:  (B, 128, 3).
            x_k:  (B, 128, 384).
        """

        batch_size = x_k.size(0)
        num_points_q = coor_q.size(1)
        num_points_k = coor_k.size(1)
        num_far = num_points_k - num_points_q
        num_dims = x_k.size(2)

        sqrdists = self.square_distance(coor_k, coor_q)
        mindist, _ = torch.topk(sqrdists, 1, dim=-1, largest=False)
        maxdist, maxidx = torch.topk(mindist, num_far, dim=1, largest=True)
        idx, _ = maxidx.squeeze(-1).sort(dim=-1)
        idx = idx.unsqueeze(1)
        idx_base = torch.arange(0, batch_size, device=x_k.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, num_far, num_dims)
        return feature

    def forward(self, shapes, key_center, key_feature):
        """
            key_feature:  (B, 32, d).
            shapes:  (B, K, 8192, 3).
            value_feature:  (B, K, 128, d).
        """
        value_center_list, value_feature_list = [], []
        for i, shape in enumerate(torch.split(shapes, 1, dim=1)):
            value_cen, value_fea, _ = self.value_encoder(shape.squeeze(1))
            value_center_list.append(value_cen)
            value_feature_list.append(value_fea)

        # process the value center
        value_center = torch.cat([i.transpose(1, 2).contiguous() for i in value_center_list], dim=1)  # B, 3*3, 128
        B, _, M = value_center.shape
        value_center = self.decrease_ctr_1(value_center).transpose(1, 2).contiguous()  # B, 128, 3
        global_value_center = torch.max(value_center, dim=1)[0]  # B 3
        value_center = torch.cat([global_value_center.unsqueeze(1).expand(-1, M, -1), value_center], dim=-1)  # B, 128, 3*2
        value_center = self.reduce_ctr_1(value_center.reshape(B * M, -1)).reshape(B, M, -1)  # B, 128, 3

        fusion_center = torch.cat([key_center, value_center], dim=1)
        ref_center = self.downsample1(fusion_center)

        # process the value feature
        value_feature = torch.cat([i.transpose(1, 2).contiguous() for i in value_feature_list], dim=1)  # B, 384*3, 128
        B, _, M = value_feature.shape
        value_feature = self.decrease_fea_1(value_feature).transpose(1, 2).contiguous()  # B, 128, 384
        global_value_feature = torch.max(value_feature, dim=1)[0]  # B 384
        value_feature = torch.cat([global_value_feature.unsqueeze(1).expand(-1, M, -1), value_feature], dim=-1)  # B, 128, 384*2
        value_feature = self.reduce_fea_1(value_feature.reshape(B * M, -1)).reshape(B, M, -1)  # B, 128, 384
        # post_value_feature = self.get_farest_feature(key_center, ref_center, value_feature)  # B, 96, 384

        # key_feature/value_feature fusion
        fusion_feature = torch.cat([key_feature, value_feature], dim=1)
        ref_feature = self.downsample2(fusion_feature)
        B, M, _ = ref_feature.shape
        # print(ref_feature.shape) #torch.Size([16, 128, 384])
        relative_xyz = self.decoder(ref_feature.reshape(B * M, -1)).reshape(B, M, -1, 3)
        recon = (relative_xyz + ref_center.unsqueeze(-2)).reshape(B, -1, 3)
        return ref_center, recon