"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 21/04/25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import pearsonr


class MLPSmall(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x, y=None):
        x = self.mlp(x)

        if y is None:
            return x

        loss = F.l1_loss(x, y)

        # compute average correlation of minibatch
        x_ = x.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()

        # r = []
        # for i in range(x_.shape[0]):
        #     r.append(pearsonr(x_[i], y_[i])[0])
        # r = torch.tensor(r).mean()

        return loss, x