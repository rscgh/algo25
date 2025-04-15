import os
root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"
print(os.getpid())

import numpy as np
from sklearn.preprocessing import StandardScaler

import os
import pickle as pk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.random_projection import johnson_lindenstrauss_min_dim
# from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import make_pipeline

from glob import glob
from tqdm.auto import tqdm


model_name="whisper-small"
#mode="2sCNKperTR-sc-srp6000" #"2sCNKperTR-last4token"
mode="2sCNKperTR-last4token"
reduction = "pca2000" # target
# outputs:
redfn= f"{acc_dir}/actv-{model_name}-algonauts_all_train-{mode}.{reduction}.feat_reduction.pkl");
actvfn= f"{acc_dir}/actv-{model_name}-algonauts_all_train-{mode}.{reduction}.npy");

# Find all the saved raw activations for all episodes
path=f"{root_data_dir}/ann_brain_data/activations/*whisper-small*{mode}.npy"
all_actvs = glob(path);
all_actvs.sort()
#all_actvs = all_actvs[:258]
print("Activation files found:", len(all_actvs), all_actvs[:3],  all_actvs[-3:])

# short function to extraxct the episode name from the filename
n_dahses= 2+model_name.count("-")
mvid = lambda x : x.split("/")[-1].split("-")[n_dahses]



##############################################################################
## Step 1: creation of the feautre reduction pipeline

features = []

sl = slice(0,-93)#slice(0,10)
for i, stim_activations_file in tqdm(enumerate(all_actvs[sl]), total=len(all_actvs[sl])):
    featd = np.load(stim_activations_file, allow_pickle=1).item()
    data = np.concatenate([v for k,v in featd.items()], axis=-1).astype(np.float32)
    #features.append(data.reshape((n,-1)))
    features.append(data)

features = np.concatenate(features, axis=0)
print("features.shape", features.shape)
features = features.reshape(features.shape[0], -1)
print("flat features.shape", features.shape)

# To further reduce the amount of data (and make sure the data fits into memory)
# we take only every 3rd TR sample
print("prev size:", features.shape)
features = features[::3,:] # to keep below 48,838
print("new size:", features.shape)

any_nan=np.isnan(features).any()
print("Any nan?", any_nan)
if any_nan:
    features = np.nan_to_num(features)

## This is the acutal feature reduction pipeline
# z-score the features
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
# features.shape  (45503, 30000)

"""
# Optional 2 steps before PCA to already break it down a bit
# use with caution, as it may lead to overfitting

# estimate the approx needed number of components
n_proj = johnson_lindenstrauss_min_dim(features.shape[0], eps=0.1)
# then downproject using SRP to reduce the number of features 
# maybe not always necessary
srp = SparseRandomProjection(n_components = n_proj, random_state=1001)
features = srp.fit_transform(features)
print(features2.shape) # (45503, 9193)

# z-score the features again because needed for PCA
scaler2 = StandardScaler()
scaler2.fit(features)
features = scaler2.transform(features2)
print(features.shape) # (45503, 9193)
"""

# do the final PCA step
n_components = 2000
pca = PCA(n_components=n_components, random_state=1001, svd_solver="full")
pca.fit(features[:, :])
print(pca.explained_variance_ratio_.sum()) #0.92546964

## save all this as a pipeline
feat_reduction = make_pipeline(scaler, pca)
#feat_reduction = make_pipeline(scaler, srp, scaler2, pca)
pk.dump(feat_reduction, open(redfn,"wb"))
print("saved:", redfn)
#feat_reduction = pk.load(open(fn,'rb'))

##############################################################################
## Step 2: Reduce ANN activations for all stimuli

features = {}
sl = slice(None)#slice(0,10)
for i, stim_activations_file in tqdm(enumerate(all_actvs[sl]), total=len(all_actvs[sl])):
    featd = np.load(stim_activations_file, allow_pickle=1).item()
    if i < 5 or (i in [100,120,300]): 
        print(i, mvid(stim_activations_file), data.shape, featd.keys(), stim_activations_file.split("/")[-1])#
    
    data = np.concatenate([v for k,v in featd.items()], axis=-1).astype(np.float32)
    n= data.shape[0]
    data = feat_reduction.transform(data)
    features[mvid(stim_activations_file)]= data

np.save(actvfn, features)
print("saved:", actvfn)