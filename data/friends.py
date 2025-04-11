"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 12/03/25
"""
import h5py
import os
import torchvision

from torch.utils.data.dataset import Dataset
from torchcodec.decoders import VideoDecoder
from glob import glob

from brainannlib.algonauts_funcs import load_fmri


class FriendsDataset(Dataset):
    def __init__(self, root, modalities=["fmri", "video"], image_transform=None, subjects=[1,2,3,5], timesample=1):
        self.root = root
        self.modalities = modalities
        self.image_transform = image_transform
        self.timesample = timesample

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

                self.fmris.append({"movie": key, "fmri": curr_fmri, "n_samples": fmri_samples})

                # Map all indexes between (tot_samples, tot_samples+fmri_samples) to current fmri index
                self.scan_idx_map.extend([last_idx] * fmri_samples)
                self.time_idx_map.extend(list(range(fmri_samples)))
                self.tot_samples += fmri_samples
                last_idx += 1

        print("Loaded", len(self.fmris), "fmri files, total samples:", self.tot_samples)

    def load_movie(self, movie_name) -> VideoDecoder:
        movie_folder = os.path.join(self.root, "algonauts_2025.competitors", "stimuli", "movies", "friends")
        season = int(movie_name[1:3])
        episode_path = os.path.join(movie_folder, f"s{season}", f"friends_{movie_name}.mkv")

        decoder = VideoDecoder(episode_path, device="cpu")
        return decoder

    def __len__(self):
        return self.tot_samples

    def __getitem__(self, idx):
        print("Getting item: ", idx)

        fmri_index = self.scan_idx_map[idx]
        print("FMRI index: ", fmri_index)

        sample_index = self.time_idx_map[idx]
        print("Sample index: ", sample_index)

        fmri = self.fmris[fmri_index]
        fmri_data = fmri["fmri"][sample_index]

        video = self.load_movie(fmri["movie"])
        n_frames = len(video)
        n_samples = fmri["n_samples"]
        print("Number of frames: ", n_frames)
        print("Number of samples: ", n_samples)

        window_length = n_frames // n_samples
        start_frame = sample_index * window_length
        end_frame = start_frame + window_length
        print("Start frame: ", start_frame)
        print("End frame: ", end_frame)
        # Ensure that the end frame does not exceed the number of frames
        if end_frame > n_frames:
            end_frame = n_frames
            print("Adjusted end frame: ", end_frame)

        print("Window length: ", window_length)

        video_data = video[start_frame:end_frame:self.timesample]  # TxCxHxW
        if self.image_transform is not None:
            video_data = self.image_transform(video_data)

        return video_data, fmri_data









