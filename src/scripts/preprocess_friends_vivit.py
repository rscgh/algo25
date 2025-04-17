"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 17/04/25
"""
import models
import torch
import argparse
import os

from torchcodec.decoders import VideoDecoder
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from transformers import VivitImageProcessor


class FriendsStimuliVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, tr=1.49, timesample=1, target_video_len=32):
        self.root = root
        self.transform = transform
        self.timesample = timesample
        self.target_video_len = target_video_len
        self.tr = tr

        self.movies = natsorted(glob(os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies/friends/**/*.mkv")))

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

        return video_data, movie_idx


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--vivit_pretrained', type=str, default="google/vivit-b-16x2-kinetics400")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    vivit_processor = VivitImageProcessor.from_pretrained(args.vivit_pretrained)

    dataset = FriendsStimuliVideoDataset(args.data_dir, transform=vivit_processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)
    print("Dataset loaded. Tot chunks:", len(dataset))

    chunk_idx = 0
    last_movie_idx = 0

    for idx, (video, movie_idx) in enumerate(tqdm(dataloader)):
        video = video['pixel_values'].squeeze(1)

        for chunk, chunk_movie_idx in zip(video, movie_idx):
            movie_name = os.path.basename(dataset.movies[chunk_movie_idx.item()])
            output_file = os.path.join(args.save_dir, f"{movie_name}_chunk_{chunk_idx:05d}.pt")
            torch.save(chunk, output_file)

            chunk_idx += 1
            if chunk_movie_idx != last_movie_idx:
                print(f"Saved {movie_name}")
                last_movie_idx = movie_idx



if __name__ == '__main__':
    main()
