"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 13/04/25
"""
import models
import torch
import argparse

from torchcodec.decoders import VideoDecoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--movie_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load the movie
    video = VideoDecoder(args.movie_path, device="cpu")
    print("Movie loaded:")

    # Load the model
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    model = models.vivit.VivitMLPContrastive(embed_dim=checkpoint['opts'].embed_dim,
                                             temperature=checkpoint['opts'].temperature)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    image_processor = model.image_processor()
    print("Model loaded")

    model.eval()
    with torch.no_grad():
        stride = 32  # can be lower if we want overlapping features
        for i in range(0, len(video), stride): # iterate over video in chunks of 32 frame and encode each chunk
            video_data = video[i:i + 32] # Vivit works with sequences of 32 frames

            # If the last chunk is smaller than 32 frames, pad it
            if len(video_data) < 32:
                # Pad with last frame
                video_data = torch.cat([video_data, video_data[-1].unsqueeze(0).expand(32 - len(video_data), -1, -1, -1)])

            # preprocess the video (it needs to be a list of tensors, one for each frame)
            lst = torch.split(video_data, 1, 0)
            lst = [l[0] for l in lst]
            video_data = image_processor(lst, return_tensors="pt").to(args.device, non_blocking=True)

            features = model.encode_video(video_data)
            print(f"({i}/{len(video)}) Video data shape: {video_data['pixel_values'].shape}, features shape: {features.shape}")
