"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 02/05/25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import VideoMAEModel, AutoImageProcessor


class VideoMAERegression(nn.Module):
    def __init__(self, pretrained="MCG-NJU/videomae-base", n_parcels=1000,
                 torch_dtype=torch.float32, criterion="mae"):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(pretrained)
        self.video_encoder = VideoMAEModel.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype
        )
        self.video_projection = nn.Linear(768, n_parcels, bias=True)
        self.criterion = criterion

    def image_processor(self):
        return self.processor

    def encode_video(self, video):
        return self.video_encoder(**video)[0][:, 0, :]

    def forward(self, video, fmri):
        video_features = self.encode_video(video)
        logits = self.video_projection(video_features)

        if len(fmri.shape) > 2:  # fmris are stacked as [bsz, 1, 1000] by dataloaders
            fmri = fmri.view(fmri.shape[0], -1)

        if self.criterion == "mae":
            loss = F.l1_loss(logits, fmri)
        elif self.criterion == "mse":
            loss = F.mse_loss(logits, fmri)

        return loss, logits

