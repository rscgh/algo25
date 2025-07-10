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
from functools import partial


def downsample_movie(movie, target, size=224):
    target_path = movie.replace(f"movies/{target}", f"movies_{size}/{target}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    command = f"ffmpeg -y -i {movie} -vf scale={size}:{size} {target_path}"

    with open(os.devnull, "wb") as devnull:
        res = subprocess.call(command.split(" "), stdout=devnull, stderr=devnull)

    if res != 0:
        print(f"Error processing {movie}")
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target', choices=["friends", "movie10", "ood"], default="friends")
    parser.add_argument('--seasons', nargs='+', default=["s5", "s6", "s7"])
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    movies = natsorted(glob(os.path.join(args.data_dir, f"algonauts_2025.competitors/stimuli/movies/{args.target}/**/*.mkv")))
    
    # hotfix to save space
    if args.target == "friends":        
        base = os.path.join(args.data_dir, f"algonauts_2025.competitors/stimuli/movies/{args.target}")
        files = []
        for sub in args.seasons:
            files.extend(glob(os.path.join(base, sub, "*.mkv")))

        movies = natsorted(files)
        #print([m.split("/")[-1] for m in movies])

    downsample_fn = partial(downsample_movie, target=args.target, size=args.size)
    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(downsample_fn, movies), total=len(movies)):
            pass


if __name__ == '__main__':
    main()
