from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

import os, re
import nibabel as nib
import numpy as np
import pickle as pk

from sklearn.preprocessing import StandardScaler
from brainannlib.stats_and_metrics import get_explained_variance
from tqdm.auto import tqdm

sc=StandardScaler();
root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"
mri_dir = f"{root_data_dir}/ann_brain_data/mri";


print(os.getpid())
n_cores=32



##############################################################
## Connectivity-matrix based

"""
# Outcomes:
* red_sample_pca91k.500comps.unscaled.pkl: expl.var: `[36, 18, 11,  5,  4,  3,  2,  2,  2,  1]`
* red_sample_pca91k.500comps.pkl: 99%, (91282, 91282) -> (500, 91282), `[36 18 11  5  4  3  2  2  2  1]`
* full_sample_pca91k.500comps.unscaled.pkl 99%, (500, 91282), `[47 17  8  5  4  3  2  2  1  1]`
* **full_sample_pca91k.500comps.scaled.pkl** 98%, (500, 91282), `[35 18 11  5  4  3  2  2  2  1]`
""";

# Load and reduce the full HCP functional connectivity matrix

file="/scratch-scc/users/robert.scholz2/HCP_Resources/GroupAvg/" + \
    "HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii"
f = nib.load(file)

cmat_z = f.get_fdata(dtype=np.float32)
#cms= cmat_z[::5] # for reduced sample
cms = np.vstack(Parallel(n_jobs=-1)(delayed(StandardScaler().fit_transform)(cmat_z[i::n_cores]) for i in range(n_cores)))
# cms.shape: (91282, 91282)

pca1 = PCA(n_components=500)
out = pca1.fit_transform(cms)

print("Comps shape:", pca1.components_.shape)
evr=(pca1.explained_variance_ratio_.round(2)[:10]*100).astype(int)
evrs=pca1.explained_variance_ratio_.sum()
print("Explained variance:", round(evrs*100), "%")
print("Explained variance ratios:", evr)
fn="full_sample_pca91k.500comps.scaled.pkl"
pk.dump(pca1, open(fn,"wb"))

"""
# outputs:
Comps shape: (500, 91282)
Explained variance: 98 %
Explained variance ratios: [35 18 11  5  4  3  2  2  2  1]
Saves: 175M full_sample_pca91k.500comps.scaled.pkl
"""



##############################################################
## Find all MRI files for each subject

from glob import glob
def find_mri_files(subj, v=False, dspath=None, episode_id=""):
    if dspath is None:
        dspath="/scratch-scc/users/robert.scholz2/cneuromod/conp-dataset/projects/cneuromod.processed"
    
    searchstring=dspath+"/fmriprep/friends/"+f"{subj}/ses-*/func/*{episode_id}space-fsLR_den-91k_bold.dtseries.nii";
    res = glob(searchstring)
    if v: print("Friends max episodes:", len(res))
    existing = [f for f in res if os.path.exists(f)]
    if v: print("Existing:", len(existing), [e.split("/")[-1] for e in existing][::10])

    find_episode_id= lambda  f: re.search(r's\d{2}e\d{2}[a-z]?', f.split("/")[-1]).group()
    friends_map = {find_episode_id(f):f for f in existing}

    # For all movie10 chunks:
    searchstring=dspath+"/fmriprep/movie10/"+f"{subj}/ses-*/func/*{episode_id}space-fsLR_den-91k_bold.dtseries.nii";
    res = glob(searchstring)
    if v: print("movie10 max episodes:", len(res))
    existing = [f for f in res if os.path.exists(f)]
    if v: print("Existing:", len(existing), [e.split("/")[-1] for e in existing][::10])
    #s="sub-01_ses-005_task-wolf13_space-fsLR_den-91k_bold.dtseries.nii"
    #s="sub-01_ses-007_task-life05_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    #re.search(r'[a-z]+\d{2}(_run-\d{1})?', s).group() # all
    find_moviepart_id = lambda f : re.search(r'[a-z]+\d{2}(_run-\d{1})?', f.split("/")[-1]).group()
    movie10_map = {find_moviepart_id(f):f for f in existing}

    friends_map.update(movie10_map)
    episodes_map = dict(sorted(friends_map.items()))
    return episodes_map;

episodes_dict = {}
for sub in ["sub-0"+str(s) for s in [1,2,3,5,6]]:
    print("##",sub)
    episodes_dict[sub] = find_mri_files(sub, v=True);

np.save(f"{acc_dir}/episodes_map_v2.npy", episodes_dict)


##############################################################
## Reduce all subject mri

n="full_sample_pca91k.500comps.scaled.pkl"
pca = pk.load(open(fn,'rb'))
episodes_dict=np.load(f"{acc_dir}/episodes_map_v2.npy",allow_pickle=1).item()

all_subs = ["sub-0"+str(s) for s in [1,2,3,5,6]]

for sub in all_subs:
    #for mvset in all_movie_sets:
    fn=f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca.npy" 
    if os.path.exists(fn): continue; 

    episodes_map=episodes_dict[sub]
    sfrmi = {}
    
    for mv in tqdm(episodes_map.keys(), desc=sub): #movie_list
        cifti = nib.load(episodes_map[mv]);
        scmri= sc.fit_transform(cifti.get_fdata(dtype=np.float32))
        embd = pca.transform(scmri);
        sfrmi[mv]=embd;

    np.save(fn, sfrmi)
    print("saved", fn)
    #!ls -ash {fn}



##############################################################
## Benchmark reconstruction


def get_joint_explained_variance(embd, pca, full, n_comps=200):
    from tqdm.auto import tqdm
    n_components= embd.shape[1]
    var_full = np.linalg.norm(full - pca.mean_)
    print(var_full)
    
    X_trans_ii = np.zeros_like(embd)
    X_trans_ii[:, :n_components] = embd[:, :n_components]
    X_approx_ii = pca.inverse_transform(X_trans_ii)
    
    var_embd = np.linalg.norm(X_approx_ii - full)
    var_ratio = 1 - (var_embd / var_full) ** 2

    return var_embd, var_full, var_ratio

# full_mri = np.concatenate([sc.fit_transform(fmri[movie]) for movie in fmri.keys()]);
#    _,_,jv200=get_joint_explained_variance(embd2, pca, full_mri, n_comps=200)    

"""
# Untested:

n="full_sample_pca91k.500comps.scaled.pkl"
pca = pk.load(open(fn,'rb'))
episodes_dict=np.load(f"{acc_dir}/episodes_map_v2.npy",allow_pickle=1).item()

all_subs = ["sub-0"+str(s) for s in [1,3,5]]

for sub in all_subs:

    episodes_map=episodes_dict[sub]
    # select only every 10th episode/run
    selected_keys = list(episodes_map.keys())[::10]

    full_mri=[]
    redu_mri=[]
    for mv in tqdm(selected_keys, desc=sub):
        fmri = nib.load(episodes_map[mv]).get_fdata(dtype=np.float32);
        embd= pca.transform(sc.fit_transform(fmri))
        full_mri.append(fmri)
        redu_mri.append(embd)

    full_mri=np.concatenate(full_mri)
    redu_mri=np.concatenate(redu_mri)
    
    # look at the first 10 components separately
    #ev=get_explained_variance(redu_mri, pca, full_mri, n_comps=10, use_tqdm=True)
    #var_embd, var_full, var_ratio = ev
    #print("Explained var:", np.round(var_ratio*100)[:10])
    
    _,_,jv=get_joint_explained_variance(redu_mri, pca, full_mri, n_comps=500)    
    print(f"Subject {sub} var explained:", jv)
"""