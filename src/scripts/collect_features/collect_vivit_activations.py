"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 14/04/25
"""
import models
import torch
import argparse
import os
import numpy as np
import logging

from torchcodec.decoders import VideoDecoder
from glob import glob
from natsort import natsorted
from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FriendsStimuliVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, tr=1.49, timesample=1, target_video_len=32, downsampled=True):
        self.root = root
        self.transform = transform
        self.timesample = timesample
        self.target_video_len = target_video_len
        self.tr = tr

        data_dir = os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies/friends/**/*.mkv")
        if downsampled:
            data_dir = os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies_224/friends/**/*.mkv")
        self.movies = natsorted(glob(data_dir))

        self.chunks = []
        self.chunk_idx_to_movie_idx = {}
        for movie_idx, movie in enumerate(self.movies):
            decoder = VideoDecoder(movie, device="cpu")
            duration = decoder.metadata.duration_seconds
            num_chunks = int(round(duration / self.tr))
            print(f"Movie {movie} has {duration:.2f} seconds and {num_chunks:.2f} chunks of {self.tr:.2f} seconds")

            # map the chunk index to the movie index
            offset = len(self.chunk_idx_to_movie_idx)

            for i in range(num_chunks):
                self.chunk_idx_to_movie_idx[i + offset] = (movie_idx, i)

            self.chunks.append(num_chunks)

        assert sum(self.chunks) == len(self.chunk_idx_to_movie_idx), f"The number of chunks ({sum(self.chunks)}) does not match the number of movies ({len(self.chunk_idx_to_movie_idx)})"

    def __len__(self):
        return sum(self.chunks)

    def __getitem__(self, idx):
        movie_idx, chunk_idx = self.chunk_idx_to_movie_idx[idx]
        movie_path = self.movies[movie_idx]

        start_t = chunk_idx * self.tr
        end_t = (chunk_idx + 1) * self.tr

        decoder = VideoDecoder(movie_path, device="cpu")
        if end_t > decoder.metadata.duration_seconds:
            end_t = decoder.metadata.duration_seconds

        chunk = decoder.get_frames_played_in_range(start_t, end_t).data

        # if the chunk is shorter than self.target_video_len, pad it
        if len(chunk) < self.target_video_len:
            # Pad with last frame
            chunk = torch.cat([chunk, chunk[-1].unsqueeze(0).expand(self.target_video_len - len(chunk), -1, -1, -1)])

        # if the chunk is longer than self.target_video_len, take last N frames
        if len(chunk) > self.target_video_len:
            chunk = chunk[-self.target_video_len:]

        lst = torch.split(chunk, 1, 0)
        lst = [l[0] for l in lst]
        video_data = self.transform(lst, return_tensors="pt")

        return video_data, movie_idx, chunk_idx


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load the model
    if args.weights:
        checkpoint = torch.load(args.weights, map_location=args.device, weights_only=False)
        model = models.vivit.VivitMLPContrastive(embed_dim=checkpoint['opts'].embed_dim,
                                                 temperature=checkpoint['opts'].temperature)
        model.load_state_dict(checkpoint['model'])
        model = model.to(args.device)
        image_processor = model.image_processor()
        print("Model loaded from", args.weights)
    else:
        model = models.vivit.VivitMLPContrastive(embed_dim=128, temperature=1.)
        image_processor = model.image_processor()
        model = model.to(args.device)
        print("Model initialized")

    dataset = FriendsStimuliVideoDataset(args.data_dir, transform=image_processor, downsampled=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)
    print("Dataset loaded. Tot chunks:", len(dataset))

    output_dir = os.path.join(os.path.dirname(args.weights), "features/friends/")
    os.makedirs(output_dir, exist_ok=True)

    prev_movie = 0
    prev_chunk = -1
    episode_features = []

    model.eval()
    for idx, (video, movie_idx, chunk_idx) in enumerate(tqdm(dataloader)):
        video = video.to(args.device, non_blocking=True)
        video['pixel_values'] = video['pixel_values'].squeeze(1)
        features = model.encode_video(video)

        for features_, movie_idx_, chunk_idx_ in zip(features, movie_idx, chunk_idx):
            if movie_idx_ != prev_movie:
                movie_name = os.path.basename(dataset.movies[movie_idx_.item()]).replace(".mkv", ".pth")
                season = int(movie_name[9:11])

                episode_features = torch.stack(episode_features, dim=0)
                season_path = os.path.join(output_dir, f"s{season}")
                os.makedirs(season_path, exist_ok=True)

                episode_path = os.path.join(season_path, movie_name)
                logging.info(f"Saving features for episode {movie_name} to: {episode_path}")

                torch.save(episode_features.cpu(), episode_path)
                episode_features = []
                prev_chunk = -1

            episode_features.append(features_)

            assert chunk_idx_ > prev_chunk
            prev_movie = movie_idx_
            prev_chunk = chunk_idx_


if __name__ == '__main__':
    main()
