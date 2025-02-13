from brainannlib.ann_activations import load_ann_embeddings
import os
import numpy as np


file_path = "/home/bagga005/algo/comp_data/ann_brain_data/activations/actv-clipvit-base-patch32-bourne01-eqsTR1.49s-5layers_clsEmbd.npy"

activations = np.load(file_path, allow_pickle=True).item()

print(activations.keys())



