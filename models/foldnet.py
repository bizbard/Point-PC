import torch
import torch.nn as nn


class Fold(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channel = config.in_channel
        self.step = config.step
        self.hidden_dim = config.hidden_dim

        a = torch.linspace(-1., 1., steps=self.step, dtype=torch.float).view(1, self.step).expand(self.step, self.step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=self.step, dtype=torch.float).view(self.step, 1).expand(self.step, self.step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.in_channel + 2, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim//2, 1),
            nn.BatchNorm1d(self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.in_channel + 3, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim//2, 1),
            nn.BatchNorm1d(self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2