
import os
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import pickle as pk

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import make_pipeline

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")


# Finding all the saved activations/embeddings from SmolLM2
mode="-last5trs"#"last500wodslast5trs"
path=f"/scratch-scc/users/robert.scholz2/ann_brain_data/activations/*SmolLM2*{mode}.npy"
mvid = lambda x : x.split("/")[-1].split("-")[3]

all_actvs = glob(path);
all_actvs.sort()
print("Found ANN activations:", len(all_actvs), all_actvs[:3], mvid(all_actvs[0]))



##############################################################################
## Step 1: creation of the feautre reduction pipeline


# Now we concatenate the activations for all friends episodes
# but spare the typical test set (e.g. season7, most of movie10)
features = []
sl = slice(0,-93)
for i, stim_activations_file in tqdm(enumerate(all_actvs[sl]), total=len(all_actvs[sl])):
    data = np.swapaxes(np.load(stim_activations_file),0,1).astype(np.float32)
    if i< 4: print(mvid(stim_activations_file), data.shape, stim_activations_file)#
    features.append(data.reshape((data.shape[0],-1)))

features = np.concatenate(features, axis=0)
print(features.shape)

# To further reduce the amount of data (and make sure the data fits into memory)
# we take only every 3rd TR sample
print("prev size:", features.nbytes / (1024**2))
features = features[::3,:] # to keep below 48,838
print("new size:", features.nbytes / (1024**2))
print("Has NaNs? ->", np.isnan(features).any())

## This is the acutal feature reduction pipeline
# z-score the features
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
print(features.shape)

# estimate the approx needed number of components
n_proj = johnson_lindenstrauss_min_dim(features.shape[0], eps=0.1)
# then downproject using SRP to reduce the number of features 
# maybe not always necessary
srp = SparseRandomProjection(n_components = n_proj, random_state=1001)
features = srp.fit_transform(features)
print(features.shape, "after SRP")

# z-score the features again because needed for PCA
scaler2 = StandardScaler()
scaler2.fit(features)
features = scaler2.transform(features)
print(features.shape)

# do the final PCA step
n_components = 2000
pca = PCA(n_components=n_components, random_state=1001, svd_solver="full")
pca.fit(features[:, :])
print("Var expl:", pca.explained_variance_ratio_.sum() )

## save all this as a pipeline
feat_reduction = make_pipeline(scaler, srp, scaler2, pca)
path= "/scratch-scc/users/robert.scholz2/ann_brain_data/activations/"
fn= os.path.join(path, f"actv-SmolLM2-1.7B-algonauts_all_train-{mode}.pca2000.feat_reduction.pkl");
#pk.dump(feat_reduction, open(fn,"wb"))



##############################################################################
## Step 2: Reduce ANN activations for all stimuli

features = {}
sl = slice(None)
for i, stim_activations_file in tqdm(enumerate(all_actvs[sl]), total=len(all_actvs[sl])):
    data = np.swapaxes(np.load(stim_activations_file),0,1).astype(np.float32)
    data= np.nan_to_num(data).reshape((data.shape[0],-1))
    data = feat_reduction.transform(data)
    features[mvid(stim_activations_file)]= data

print(features.keys(), features["s02e01a"].shape);

fn= os.path.join(path, f"actv-SmolLM2-1.7B-algonauts_all_train-last5trs.pca2000.npy");
np.save(fn, features)



"""
# Reduced features can now be loaded as:

# load the language features (tutorial had: 250)
fn= os.path.join(acc_dir, f"actv-SmolLM2-1.7B-algonauts_all_train-last5trs.pca2000.npy");
lang_features = np.load(fn, allow_pickle=True).item()
lang_features = {k.split("_")[-1]: v for k,v in lang_features.items()}
lang_features = {k:v[:,:] for k,v in lang_features.items()}
print("language:", lang_features["s01e01a"].shape)
""";