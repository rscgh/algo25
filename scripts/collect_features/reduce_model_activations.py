
import os
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import pickle as pk

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"


#########################################################################
# BEATsIter3p - 4 Layers, Raw activations for every 10th token
#########################################################################

# find the activation file collected for each of the movie clips
path=f"/scratch-scc/users/robert.scholz2/ann_brain_data/activations/*BEATs*raw.npy"
mvid = lambda x : x.split("/")[-1].split("-")[2]
all_actvs = glob(path);
all_actvs.sort()
print("Found ANN activations:", len(all_actvs), all_actvs[:3], mvid(all_actvs[0]))

###---------------------------------------------------------------------
## Step 1: creation of the feautre reduction pipeline

# Now we concatenate the activations for all friends episodes
# but spare the typical test set (e.g. season7, most of movie10)
features = []
sl = slice(0,-93)
for i, stim_activations_file in tqdm(enumerate(all_actvs[sl]), total=len(all_actvs[sl])):
    x=np.load(stim_activations_file, allow_pickle=1).item()
    # x[layer_name] has shape for the first clip: (405, 11, 768)
    data = np.concatenate([x[k].reshape((x[k].shape[0],-1)) for k in x.keys()], axis=1)
    data = data.astype(np.float32);
    if i< 10: print(mvid(stim_activations_file), data.shape)
    # e.g. bourne01 (405, 33792)
    features.append(data)

features = np.concatenate(features, axis=0)
print(features.shape)

# To further reduce the amount of data (and make sure the data fits into memory)
# we take only every 3rd TR sample
print("prev size:", features.nbytes / (1024**2))
features = features[::3,:] # to keep below 48,838
print("new size:", features.nbytes / (1024**2))
print("Has NaNs? ->", np.isnan(features).any()) # no

## This is the acutal feature reduction pipeline
# z-score the features
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
print(features.shape) # (39420, 33792)

# do the final PCA step
# reducing (39420, 33792) to 2000 takes 30mins with randomized solver
n_components = 2000
pca = PCA(n_components=n_components, random_state=1001, svd_solver="randomized")
pca.fit(features[:, :])
print("Var expl:", pca.explained_variance_ratio_.sum() )

## save all this as a pipeline
feat_reduction = make_pipeline(scaler, pca)
fn= os.path.join(acc_dir, f"actv-BEATsIter3p-algonauts_all_train-4LrawEv10tok.pca2000.feat_reduction.pkl");
pk.dump(feat_reduction, open(fn,"wb"))

###---------------------------------------------------------------------
## Step 2: apply it to all clips/episodes

features = {}
sl = slice(None)
for i, stim_activations_file in tqdm(enumerate(all_actvs[sl]), total=len(all_actvs[sl])):
    x=np.load(stim_activations_file, allow_pickle=1).item()
    data = np.concatenate([x[k].reshape((x[k].shape[0],-1)) for k in x.keys()], axis=1)
    data = data.astype(np.float32);
    data = feat_reduction.transform(data)
    features[mvid(stim_activations_file)]= data

print(features.keys(), features["friends_s02e01a"].shape);
fn= os.path.join(acc_dir, f"actv-BEATsIter3p-algonauts_all_train-4LrawEv10tok.pca2000.npy");
np.save(fn, features)


#########################################################################
# dinov2Lftimagenet1k - 4 Layers, only the CLS token per layer
#########################################################################
# the CLS token may only be actually meaningfull in the last layer

# Layers: ['dinov2.encoder.layer.0', 'dinov2.encoder.layer.8', 
#       'dinov2.encoder.layer.15', 'dinov2.encoder.layer.23'])

"""
# INdividual clip activations shape
dinov2.encoder.layer.0 (405, 1024)
dinov2.encoder.layer.8 (405, 1024)
dinov2.encoder.layer.15 (405, 1024)
dinov2.encoder.layer.23 (405, 1024)
""";