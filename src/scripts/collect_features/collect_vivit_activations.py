"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 14/04/25
"""
import models
import torch
import argparse
import os

from torchcodec.decoders import VideoDecoder
from glob import glob
from natsort import natsorted
from tqdm import tqdm


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

        decoder = VideoDecoder(movie_path, device="cpu")
        start_t = chunk_idx * self.tr
        end_t = (chunk_idx + 1) * self.tr
        chunk = decoder.get_frames_played_in_range(start_t, end_t).data

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

    dataset = FriendsStimuliVideoDataset(args.data_dir, transform=image_processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)
    print("Dataset loaded. Tot chunks:", len(dataset))

    model.eval()
    for idx, (video, movie_idx) in enumerate(tqdm(dataloader)):
        video = video.to(args.device, non_blocking=True)
        features = model.encode_video(video)

        print(f"({idx}/{len(dataloader)}) Video data shape: {video['pixel_values'].shape}, features shape: {features.shape}")

if __name__ == '__main__':
    main()
