"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 14/04/25
"""
import models
import torch
import torch.multiprocessing as mp
import argparse
import os
import numpy as np
import logging

from torchcodec.decoders import VideoDecoder
from glob import glob
from natsort import natsorted
from tqdm import tqdm

from data.friends import load_video_chunk

# mp.set_start_method("spawn", force=True)

TORCHCODEC_DEVICE = "cpu"

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FriendsStimuliVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, tr=1.49, timesample=1, target_video_len=32, downsampled=True,
                 stimulus_window=1, hrf_delay=0):
        self.root = root
        self.transform = transform
        self.timesample = timesample
        self.target_video_len = target_video_len
        self.tr = tr
        self.stimulus_window = stimulus_window
        self.hrf_delay = hrf_delay

        data_dir = os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies/friends/**/*.mkv")
        if downsampled:
            data_dir = os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies_224/friends/**/*.mkv")
        self.movies = natsorted(glob(data_dir))

        self.chunks = []
        self.chunk_idx_to_movie_idx = {}
        for movie_idx, movie in enumerate(self.movies):
            decoder = VideoDecoder(movie, device=TORCHCODEC_DEVICE)
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

        video_data = load_video_chunk(movie_path, chunk_idx, self.tr, self.target_video_len, self.transform,
                                      stimulus_window=self.stimulus_window, hrf_delay=self.hrf_delay)
        return video_data, movie_idx, chunk_idx


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--output_dir', type=str, help="output directory (only if weights is None)")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--downsampled', action='store_true', help="Use downsampled videos (224x224)")
    parser.add_argument('--amp', action='store_true', help="Use automatic mixed precision")
    parser.add_argument('--stimulus_window', type=int, default=1)
    parser.add_argument('--hrf_delay', type=int, default=0)
    args = parser.parse_args()

    # Load the model
    if args.weights:
        checkpoint = torch.load(args.weights, map_location=args.device, weights_only=False)
        model = models.vivit.VivitMLPContrastive(embed_dim=checkpoint['opts'].embed_dim,
                                                 temperature=checkpoint['opts'].temperature,
                                                 fmri_window=checkpoint['opts'].fmri_window,)
        model.load_state_dict(checkpoint['model'])
        model = model.to(args.device)
        image_processor = model.image_processor()
        print("Model loaded from", args.weights)
    else:
        model = models.vivit.VivitMLPContrastive(embed_dim=128, temperature=1.)
        image_processor = model.image_processor()
        model = model.to(args.device)
        print("Model initialized")

    dataset = FriendsStimuliVideoDataset(args.data_dir, transform=image_processor,
                                         downsampled=args.downsampled,
                                         stimulus_window=args.stimulus_window,
                                         hrf_delay=args.hrf_delay)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    print("Dataset loaded. Tot chunks:", len(dataset))

    output_dir = f"features_w{args.stimulus_window}_hrf{args.hrf_delay}/friends/"
    if args.weights:
        output_dir = os.path.join(os.path.dirname(args.weights), output_dir)
    else:
        output_dir = os.path.join(args.output_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    curr_movie_idx = 0
    curr_movie_name = os.path.basename(dataset.movies[0].replace(".mkv", ".pth"))
    prev_chunk = -1
    episode_features = []

    model.eval()
    for idx, (video, movie_idx, chunk_idx) in enumerate(tqdm(dataloader)):
        video = video.to(args.device, non_blocking=True)
        video['pixel_values'] = video['pixel_values'].squeeze(1)

        with torch.amp.autocast("cuda", enabled=args.amp):
            features = model.encode_video(video)

        for features_, movie_idx_, chunk_idx_ in zip(features, movie_idx, chunk_idx):
            if movie_idx_ != curr_movie_idx:
                print(curr_movie_name, curr_movie_name[9:11])
                season = int(curr_movie_name[9:11])

                episode_features = torch.stack(episode_features, dim=0)
                season_path = os.path.join(output_dir, f"s{season}")
                os.makedirs(season_path, exist_ok=True)

                episode_path = os.path.join(season_path, curr_movie_name)
                logging.info(f"Saving features for episode {curr_movie_name} to: {episode_path} (shape: {episode_features.shape})")

                torch.save(episode_features.cpu(), episode_path)
                episode_features = []
                prev_chunk = -1
                curr_movie_name = os.path.basename(dataset.movies[movie_idx_.item()]).replace(".mkv", ".pth")


            episode_features.append(features_)

            assert chunk_idx_ > prev_chunk
            curr_movie_idx = movie_idx_
            prev_chunk = chunk_idx_


if __name__ == '__main__':
    main()
