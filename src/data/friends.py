"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 12/03/25
"""
import h5py
import os
import torch
import torchvision
import logging
import numpy as np

from torch.utils.data.dataset import Dataset
from torchcodec.decoders import VideoDecoder
from glob import glob

from brainannlib.algonauts_funcs import load_fmri


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def load_video_chunk(path, sample_index, tr=1.49, target_video_len=32,
                     transform=None, stimulus_window=1, hrf_delay=0, device="cpu"):

    # compute the starting index of the current stimulus,
    # automatically adjusts for out of bounds windows
    # as long as target_video_len < tr*fps (~44)
    start_index = max(0, sample_index - stimulus_window - hrf_delay + 1)
    start_t = start_index * tr
    end_t = (start_index + 1) * tr

    decoder = VideoDecoder(path, device=device)
    if end_t > decoder.metadata.duration_seconds:
        end_t = decoder.metadata.duration_seconds

    chunk = decoder.get_frames_played_in_range(start_t, end_t).data

    # if the chunk is shorter than self.target_video_len, pad it
    if len(chunk) < target_video_len:
        # Pad with last frame
        chunk = torch.cat([chunk, chunk[-1].unsqueeze(0).expand(target_video_len - len(chunk), -1, -1, -1)])

    # if the chunk is longer than self.target_video_len, take N frames uniformly
    if len(chunk) > target_video_len:
        idx = np.linspace(0, len(chunk) - 1, target_video_len).astype(int)
        chunk = chunk[idx]

    if transform is not None:
        lst = torch.split(chunk, 1, 0)
        lst = [l[0] for l in lst]
        chunk = transform(lst, return_tensors="pt")

    return chunk


class FriendsDataset(Dataset):
    def __init__(self, root, modalities=["fmri", "video"], image_transform=None, tr=1.49, seasons=[1,2,3,4,5,6],
                 subjects=[1,2,3,5], timesample=1, target_video_len=32, stimulus_window=1, hrf_delay=0,
                 downsampled=False, fmri_window=1):
        self.root = root
        self.modalities = modalities
        self.seasons = seasons
        self.image_transform = image_transform
        self.timesample = timesample
        self.target_video_len = target_video_len
        self.downsampled = downsampled
        self.tr = tr
        self.stimulus_window = stimulus_window
        self.hrf_delay = hrf_delay
        self.subjects = subjects

        if fmri_window % 2 == 0:
            raise ValueError("fmri_window must be odd, got {}".format(fmri_window))
        self.fmri_window = fmri_window // 2


        # List all .h5 files in the root directory
        self.fmris = []
        self.tot_samples = 0
        self.scan_idx_map = []
        self.time_idx_map = []

        last_idx = 0
        for subject in subjects:
            fmri = load_fmri(root, subject, targets=["friends"])

            for key in fmri.keys():
                curr_fmri = fmri[key]
                fmri_samples = curr_fmri.shape[0]

                season = int(key[1:3])
                if season not in self.seasons:
                    continue

                self.fmris.append({"movie": key, "fmri": curr_fmri, "n_samples": fmri_samples})

                # Map all indexes between (tot_samples, tot_samples+fmri_samples) to current fmri index
                self.scan_idx_map.extend([last_idx] * fmri_samples)
                self.time_idx_map.extend(list(range(fmri_samples)))
                self.tot_samples += fmri_samples
                last_idx += 1

        print("Loaded", len(self.fmris), "fmri files, total samples:", self.tot_samples)

    def get_movie_path(self, movie_name) -> VideoDecoder:
        movie_folder = os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies/friends")
        if self.downsampled:
            movie_folder = os.path.join(self.root, "algonauts_2025.competitors/stimuli/movies_224/friends")

        season = int(movie_name[1:3])
        episode_path = os.path.join(movie_folder, f"s{season}", f"friends_{movie_name}.mkv")
        return episode_path

    def __len__(self):
        return self.tot_samples

    def __getitem__(self, idx):
        fmri_index = self.scan_idx_map[idx]
        sample_index = self.time_idx_map[idx]

        fmri = self.fmris[fmri_index]
        fmri_data = None

        if self.fmri_window == 0:
            fmri_data = torch.tensor(fmri["fmri"][sample_index])

        elif sample_index - self.fmri_window - 1 >= 0 and sample_index + self.fmri_window < fmri["n_samples"]:
            fmri_data = torch.tensor(fmri["fmri"][sample_index - self.fmri_window - 1:sample_index + self.fmri_window])

        elif sample_index - self.fmri_window - 1 < 0:
            fmri_data = torch.tensor(fmri["fmri"][:sample_index + self.fmri_window])
            # print("Padding with first frame:", fmri_data.shape, fmri_data[0].shape, sample_index, self.fmri_window)
            # pad with the first frame at the beginning to reach self.fmri_window
            pad = fmri_data[0].unsqueeze(0).repeat(self.fmri_window + 1 - sample_index, 1)
            fmri_data = torch.cat([pad, fmri_data], dim=0)

        elif sample_index + self.fmri_window >= fmri["n_samples"]:
            fmri_data = torch.tensor(fmri["fmri"][sample_index - self.fmri_window - 1:])
            # print("Padding with last frame:", fmri_data.shape, fmri_data[0].shape, sample_index, self.fmri_window)
            # pad with the last frame at the end to reach self.fmri_window
            pad = fmri_data[-1].unsqueeze(0).repeat(sample_index + self.fmri_window - fmri["n_samples"], 1)
            fmri_data = torch.cat([fmri_data, pad], dim=0)

        assert fmri_data is not None, f"fmri_data is None for sample {sample_index} in movie {fmri['movie']}"

        movie_path = self.get_movie_path(fmri["movie"])
        movie_chunk = load_video_chunk(movie_path, sample_index, tr=self.tr,
                                       target_video_len=self.target_video_len,
                                       transform=self.image_transform,
                                       stimulus_window=self.stimulus_window,
                                       hrf_delay=self.hrf_delay)
        return movie_chunk, fmri_data


class FriendsFeatureDataset(Dataset):
    def __init__(self, root, features_root, subjects=[1,2,3,5], seasons=[1,2,3,4,5,6],
                 stimulus_window=3):
        self.root = root
        self.features_root = features_root
        self.seasons = seasons
        self.subjects = subjects
        self.stimulus_window = stimulus_window

        # List all .h5 files in the root directory
        self.fmris = []
        self.tot_samples = 0
        self.scan_idx_map = []
        self.time_idx_map = []

        last_idx = 0
        for subject in subjects:
            fmri = load_fmri(root, subject, targets=["friends"])

            for key in fmri.keys():
                curr_fmri = fmri[key]
                fmri_samples = curr_fmri.shape[0]

                season = int(key[1:3])
                if season not in self.seasons:
                    continue

                self.fmris.append({"movie": key, "fmri": curr_fmri, "n_samples": fmri_samples, "subject": subject})

                # Map all indexes between (tot_samples, tot_samples+fmri_samples) to current fmri index
                self.scan_idx_map.extend([last_idx] * fmri_samples)
                self.time_idx_map.extend(list(range(fmri_samples)))
                self.tot_samples += fmri_samples
                last_idx += 1

        print("Loaded", len(self.fmris), "fmri files, total samples:", self.tot_samples)

    def load_movie_features(self, movie_name) -> torch.Tensor:
        movie_folder = os.path.join(self.features_root, "friends")

        season = int(movie_name[1:3])
        episode_path = os.path.join(movie_folder, f"s{season}", f"friends_{movie_name}.pth")

        features = torch.load(episode_path, map_location="cpu")
        return features

    def __len__(self):
        return self.tot_samples

    def __getitem__(self, idx):
        fmri_index = self.scan_idx_map[idx]
        sample_index = self.time_idx_map[idx]

        fmri = self.fmris[fmri_index]

        if sample_index >= fmri["n_samples"]:
            fmri_data = fmri["fmri"][-1]
        else:
            fmri_data = fmri["fmri"][sample_index]

        features = self.load_movie_features(fmri["movie"])

        # logging.info(f"Loaded features for {fmri['movie']} with shape {features.shape} for subject {fmri['subject']}")

        # assert len(features) == fmri["n_samples"], f"Features length {len(features)} does not match fmri samples {fmri['n_samples']}"

        # get the features for the current sample in the range sample_index-self.stimulus_window:sample_index
        if sample_index - self.stimulus_window < 0:
            # pad with the first sample to reach self.stimulus_window
            pad = torch.zeros(self.stimulus_window - sample_index, features.shape[1], device=features.device)
            features = torch.cat([pad, features[:sample_index]], dim=0)
        elif sample_index >= features.shape[0]:
            features = features[-1 - self.stimulus_window:-1]
        else:
            features = features[sample_index - self.stimulus_window:sample_index]

        return features.flatten(), fmri_data



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/barbano/data")
    parser.add_argument('--features_dir', type=str)
    args = parser.parse_args()

    dataset = FriendsFeatureDataset(root=args.data_dir, features_root=args.features_dir)
    sample = dataset[0]
    print(sample[0].shape, sample[1].shape)



