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
                 temperature=1.0, torch_dtype=torch.float32):
        super().__init__()

        self.vivit_processor = VivitImageProcessor.from_pretrained(vivit_pretrained)
        self.video_encoder = VivitModel.from_pretrained(
            vivit_pretrained,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype
        )
        self.video_projection = nn.Linear(768, embed_dim, bias=False)

        self.fmri_encoder = torchvision.ops.MLP(
            in_channels=1000,
            hidden_channels=[512, embed_dim*2, embed_dim],
            activation_layer=torch.nn.ReLU,
            bias=True,
            dropout=0
        )

        self.temperature = temperature


    def image_processor(self):
        return self.vivit_processor


    def forward(self, video, fmri):
        video_features = self.video_encoder(**video)
        video_features = self.video_projection(video_features.pooler_output)

        fmri_features = self.fmri_encoder(fmri)

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
