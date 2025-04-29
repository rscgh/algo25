"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 11/04/25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers import VivitModel, VivitImageProcessor


class VivitMLPContrastive(nn.Module):
    def __init__(self, vivit_pretrained="google/vivit-b-16x2-kinetics400", embed_dim=128,
                 temperature=1.0, fmri_window=1, torch_dtype=torch.float32):
        super().__init__()

        self.vivit_processor = VivitImageProcessor.from_pretrained(vivit_pretrained)
        self.video_encoder = VivitModel.from_pretrained(
            vivit_pretrained,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype
        )
        self.video_projection = nn.Linear(768, embed_dim, bias=False)

        self.fmri_encoder = torchvision.ops.MLP(
            in_channels=1000*fmri_window,
            hidden_channels=[512*fmri_window, embed_dim*2, embed_dim],
            activation_layer=torch.nn.ReLU,
            bias=True,
            dropout=0
        )

        self.temperature = temperature


    def image_processor(self):
        return self.vivit_processor

    def encode_video(self, video):
        return self.video_encoder(**video)[0][:, 0, :]

    def encode_fmri(self, fmri):
        if len(fmri.shape) > 2:
            fmri = fmri.reshape(fmri.shape[0], fmri.shape[1]*fmri.shape[2])
        fmri_features = self.fmri_encoder(fmri)
        return fmri_features

    def forward(self, video, fmri):
        video_features = self.video_encoder(**video)
        video_features = self.video_projection(video_features[0][:, 0, :])

        fmri_features = self.encode_fmri(fmri)

        video_features = F.normalize(video_features, dim=-1)
        fmri_features = F.normalize(fmri_features, dim=-1)

        logits_video = (video_features @ fmri_features.t()) / self.temperature
        logits_fmri = logits_video.t()

        batch_size = logits_video.shape[0]
        labels = torch.arange(batch_size, device=logits_video.device).long()
        loss = (
            F.cross_entropy(logits_video, labels) +
            F.cross_entropy(logits_fmri, labels)
        ) / 2
        return loss


class VivitConvContrastive(VivitMLPContrastive):
    def __init__(self, vivit_pretrained="google/vivit-b-16x2-kinetics400", embed_dim=128,
                 temperature=1.0, fmri_window=1, torch_dtype=torch.float32):
        super().__init__(vivit_pretrained, embed_dim, temperature, fmri_window, torch_dtype)

        self.fmri_encoder = nn.Sequential(
            nn.Conv1d(1000, 1000, kernel_size=3, stride=1, padding=0, groups=1000),
            nn.ReLU(),
            nn.Conv1d(1000, 1000, kernel_size=3, stride=1, padding=0, groups=1000),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_fmri(self, fmri):
        fmri_features = self.fmri_encoder(fmri.permute(0, 2, 1))
        return fmri_features