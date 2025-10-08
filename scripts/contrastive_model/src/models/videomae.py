"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 02/05/25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import VideoMAEModel, AutoImageProcessor


_SUBJECT_EMBEDDING_SIZE = 10

class VideoMAERegression(nn.Module):
    def __init__(self, pretrained="MCG-NJU/videomae-base", n_parcels=1000,
                 torch_dtype=torch.float32, criterion="mae", freeze_encoder=False, use_subject_id=False):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(pretrained)
        self.video_encoder = VideoMAEModel.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype
        )
        self.video_projection = nn.Linear(768 + _SUBJECT_EMBEDDING_SIZE if use_subject_id else 0, n_parcels, bias=True)
        self.criterion = criterion

        if freeze_encoder:
            for param in self.video_encoder.parameters():
                param.requires_grad = False

        self.subject_embedding = nn.Linear(1, _SUBJECT_EMBEDDING_SIZE) # nn.Embedding(1, _SUBJECT_EMBEDDING_SIZE)

    def image_processor(self):
        return self.processor

    def encode_video(self, video):
        return self.video_encoder(**video)[0][:, 0, :]

    def predict_activation(self, video_features, subject_id=None):
        if subject_id is not None:
            subject_embedding = self.subject_embedding(subject_id[:, None])
            video_features = torch.cat((video_features, subject_embedding), dim=1)

        logits = self.video_projection(video_features)
        return logits


    def forward(self, video, fmri, subject_id=None):
        video_features = self.encode_video(video)
        logits = self.predict_activation(video_features, subject_id)

        if len(fmri.shape) > 2:  # fmris are stacked as [bsz, 1, 1000] by dataloaders
            fmri = fmri.view(fmri.shape[0], -1)

        if self.criterion == "mae":
            loss = F.l1_loss(logits, fmri)
        elif self.criterion == "mse":
            loss = F.mse_loss(logits, fmri)

        return loss, logits
