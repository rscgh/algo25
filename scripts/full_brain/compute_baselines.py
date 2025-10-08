import os, re

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"
mri_dir = f"{root_data_dir}/ann_brain_data/mri";

import numpy as np
import pickle as pk

from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from brainannlib.stats_and_metrics import corr_score

# Load the MRI data

fmri_dict = {}
all_subs = ["sub-0"+str(s) for s in [1,2,3,5,6]]
for sub in all_subs:
    fn=f"{mri_dir}/{sub}_allmovies_gcmpca.npy" 
    subj_fmri = np.load(fn, allow_pickle=1).item()
    # only use the first run for figures and life
    #subj_fmri = {k.replace("_run-1",""): v for k, v in subj_fmri.items() if not k.endswith("_run-2")};
    fmri_dict[sub]=subj_fmri
    print(sub, len(fmri_dict[sub].keys()))

fn="/scratch-scc/users/robert.scholz2/full_sample_pca91k.500comps.scaled.pkl"
pca = pk.load(open(fn,'rb'))

fmri = fmri_dict;
ess,ese=5,5 # excluded_samples_start, excluded_samples_end

# find all the episodes for the two repeated movies; per repetition run
life1 = [k for k in fmri["sub-01"].keys() if ("life" in k) and ("_run-1" in k)]
life2 = [k for k in fmri["sub-01"].keys() if ("life" in k) and ("_run-2" in k)]
figs1 = [k for k in fmri["sub-01"].keys() if ("figures" in k) and ("_run-1" in k)]
figs2 = [k for k in fmri["sub-01"].keys() if ("figures" in k) and ("_run-2" in k)]
print(life1)

# Check the run lengths (should be similiar)
for s in [1,2,3,5,6]:
    print(s, "life1:",[fmri[sub][epi].shape[0] for epi in life1])
    print(s,"life2:",[fmri[sub][epi].shape[0] for epi in life2])
    print(s,"figs1:",[fmri[sub][epi].shape[0] for epi in figs1])
    print(s,"figs2:",[fmri[sub][epi].shape[0] for epi in figs2])


#################################################################################################
## Full brain test retest (Correlation)

from collections import defaultdict
scores_pc=defaultdict(list)
scores_bw=defaultdict(list)

for s in [1,2,3,5,6]:
    sub=f"sub-0{s}"

    test, retst=[], [];
    for epi1,epi2 in zip(life1,life2):
        t =fmri[sub][epi1]
        r =fmri[sub][epi2]
        # ensure that we compare runs of the same length
        n = min(t.shape[0], r.shape[0])
        test.append(t[:n][ess:-ese])
        retst.append(r[:n][ess:-ese]);

    test = np.concatenate(test)
    retst = np.concatenate(retst)
    #ea1, mea1 = compute_encoding_accuracy(test, retst, s, "test-retest")
    score_pc1=corr_score(test, retst)
    fulla = pca.inverse_transform(test)
    fullb = pca.inverse_transform(retst)
    with np.errstate(invalid='ignore'): # because of medial wall ==0
          score_bw1=corr_score(fulla, fullb) 
    print("test-retest-life", sub+":\t", score_bw1.mean().round(3));

    test, retst=[], [];
    for epi1,epi2 in zip(figs1,figs2):
        t =fmri[sub][epi1]
        r =fmri[sub][epi2]
        # ensure that we compare runs of the same length
        n = min(t.shape[0], r.shape[0])
        test.append(t[:n][ess:-ese])
        retst.append(r[:n][ess:-ese]);

    test = np.concatenate(test)
    retst = np.concatenate(retst)

    #ea2, mea2 = compute_encoding_accuracy(test, retst, s, "test-retest")
    score_pc2=corr_score(test, retst)
    fulla = pca.inverse_transform(test)
    fullb = pca.inverse_transform(retst)
    with np.errstate(invalid='ignore'): # because of medial wall ==0
          score_bw2=corr_score(fulla, fullb) 
    print("test-retest-figures", sub+":\t", score_bw2.mean().round(3));

    scores_pc["life"].append(score_pc1)
    scores_pc["figures"].append(score_pc2)
    scores_bw["life"].append(score_bw1)
    scores_bw["figures"].append(score_bw2)

fn=f"{acc_dir}/retest_bw+pca_baseline.npy"
np.save(fn, dict(scores_pc=scores_pc, scores_bw=scores_bw, subj_nrs=[1,3,5]));

"""
test-retest-life sub-01:	 0.206
test-retest-figures sub-01:	 0.323
test-retest-life sub-02:	 0.176
test-retest-figures sub-02:	 0.255
test-retest-life sub-03:	 0.266
test-retest-figures sub-03:	 0.32
test-retest-life sub-05:	 0.172
test-retest-figures sub-05:	 0.319
test-retest-life sub-06:	 0.172
test-retest-figures sub-06:	 0.237
"""


#################################################################################################
## Full brain test-retest R2

# own clean, and motivated derivation
def compute_noise_ceiling(y1, y2):
    """
    y1, y2: arrays of shape (n_samples, n_variables)
    Returns:
        noise_ceiling: array of shape (n_variables,)
    """
    y_avg = (y1 + y2) / 2
    var_diff = np.var(y1 - y2, axis=0, ddof=1)
    var_y1 = np.var(y1, axis=0, ddof=1)
    var_avg = np.var(y_avg, axis=0, ddof=1)

    noise_ceiling = (var_avg - 0.25 * var_diff) / var_y1
    return noise_ceiling

score_fn = compute_noise_ceiling # corr_score


scores_pc=defaultdict(list)
scores_bw=defaultdict(list)


sc = StandardScaler()

for s in [1,2,3,5,6]:
    sub=f"sub-0{s}"

    test, retst=[], [];
    for epi1,epi2 in zip(life1,life2):
        t =fmri[sub][epi1]
        r =fmri[sub][epi2]
        # ensure that we compare runs of the same length
        n = min(t.shape[0], r.shape[0])
        tt=sc.fit_transform(t[:n][ess:-ese])
        rr=sc.fit_transform(r[:n][ess:-ese])
        test.append(tt)
        retst.append(rr);
        

    test = np.concatenate(test)
    retst = np.concatenate(retst)
    score_pc1=score_fn(test, retst)
    fulla = pca.inverse_transform(test)
    fullb = pca.inverse_transform(retst)
    with np.errstate(invalid='ignore'): # because of medial wall ==0
          score_bw1=score_fn(fulla, fullb) 
    print("test-retest-life", sub+":\t", score_bw1.mean().round(3));

    test, retst=[], [];
    for epi1,epi2 in zip(figs1,figs2):
        t =fmri[sub][epi1]
        r =fmri[sub][epi2]
        # ensure that we compare runs of the same length
        n = min(t.shape[0], r.shape[0])
        tt=t[:n][ess:-ese] # sc.fit_transform(t[:n][ess:-ese])
        rr=r[:n][ess:-ese] # sc.fit_transform(r[:n][ess:-ese])
        test.append(tt)
        retst.append(rr);
        
    test = np.concatenate(test)
    retst = np.concatenate(retst)

    #ea2, mea2 = compute_encoding_accuracy(test, retst, s, "test-retest")
    score_pc2=score_fn(test, retst)
    fulla = pca.inverse_transform(test)
    fullb = pca.inverse_transform(retst)
    with np.errstate(invalid='ignore'): # because of medial wall ==0
          score_bw2=score_fn(fulla, fullb) 
    print("test-retest-figures", sub+":\t", score_bw2.mean().round(3));

    scores_pc["life"].append(score_pc1)
    scores_pc["figures"].append(score_pc2)
    scores_bw["life"].append(score_bw1)
    scores_bw["figures"].append(score_bw2)


fn=f"{acc_dir}/retest_bw+pca_baseline.R2.npy"
np.save(fn, dict(scores_pc=scores_pc, scores_bw=scores_bw, subj_nrs=[1,3,5]));

"""
test-retest-life sub-01:	 0.218
test-retest-figures sub-01:	 0.322
test-retest-life sub-02:	 0.154
test-retest-figures sub-02:	 0.238
test-retest-life sub-03:	 0.257
test-retest-figures sub-03:	 0.307
test-retest-life sub-05:	 0.151
test-retest-figures sub-05:	 0.328
test-retest-life sub-06:	 0.159
test-retest-figures sub-06:	 0.24
"""
