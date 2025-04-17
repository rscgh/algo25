"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 17/04/25
"""
import argparse
import os
import multiprocessing
import subprocess

from glob import glob
from natsort import natsorted
from tqdm import tqdm


def downsample_movie(movie):
    target_path = movie.replace("movies/friends", "movies/friends_224")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    command = f"ffmpeg -y -i {movie} -vf scale=224:224 {target_path}"

    with open(os.devnull, "wb") as devnull:
        res = subprocess.call(command.split(" "), stdout=devnull, stderr=devnull)

    if res != 0:
        print(f"Error processing {movie}")
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()

    movies = natsorted(glob(os.path.join(args.data_dir, "algonauts_2025.competitors/stimuli/movies/friends/**/*.mkv")))

    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(downsample_movie, movies), total=len(movies)):
            pass


if __name__ == '__main__':
    main()
