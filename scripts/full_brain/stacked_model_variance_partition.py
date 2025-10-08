import os
import numpy as np
import pickle as pk

from brainannlib.stats_and_metrics import R2
from brainannlib.algonauts_funcs import (
    align_features_and_fmri_samples, 
)

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
featred_dir = f"{root_data_dir}/ann_brain_data/featred"
res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"

import os
root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"

###########################################################################
# Stacked regression functions

def train_and_predict(enc_features, fmri, model_id, sub, movies_train, movies_optim=["friends-s06"], movies_test = ["figures"],use_prefit_model=False):
    #model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{feature_set}.s1to5.{sub}.pkl"

    # we have: x_train for iniial per modality model fitting
    # we have: x_optim for stacked model estimation
    # and:     x_test for evaluation of the stacked model
    
    enc_features = {"visual": enc_features}
    
    x_optim, y_optim = align_features_and_fmri_samples(
        enc_features, fmri, ess, ese, hrf_delay, stimulus_window, movies_optim, n_targets=500)

    if use_prefit_model:
        model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{model_id}.s1to5.{sub}.avgmodel.pkl"
        model = pk.load(open(model_fn,'rb'))
    else:
        from brainannlib.algonauts_funcs import train_sklearn_ridgecv
        x_train, y_train = align_features_and_fmri_samples(
            enc_features, fmri, ess, ese, hrf_delay, stimulus_window, movies_train, n_targets=500)
        print("fitting new model")
        model = train_sklearn_ridgecv(x_train, y_train)

    x_test, y_test = align_features_and_fmri_samples(enc_features, fmri, ess, 
        ese, hrf_delay, stimulus_window, movies_test, n_targets=500)

    y_optim_pred = model.predict(x_optim);
    y_test_pred = model.predict(x_test);

    set_err = y_optim - y_optim_pred
    return set_err, y_optim_pred, y_optim, y_test, y_test_pred    

def esimate_stacked_model_coeffs(err):
    # intialize a few matrices
    n_feature_sets=len(err)
    q = matrix(np.zeros((n_feature_sets)))
    G = matrix(-np.eye(n_feature_sets, n_feature_sets))
    h = matrix(np.zeros(n_feature_sets))
    A = matrix(np.ones((1, n_feature_sets)))
    b = matrix(np.ones(1))

    # some kind of error covariance matrix (?)
    # aka the nly real input into our stacked model estimation
    P = np.zeros((n_vox, n_feature_sets, n_feature_sets))
    for i in range(n_feature_sets):
        for j in range(n_feature_sets):
            P[:, i, j] = np.mean(err[i] * err[j], 0)

    # The output weight matrix (to be used for stacked predictions)
    S = np.zeros((n_vox, n_feature_sets)) 

    for i in range(0, n_vox):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel (solve "the quadratic programming problem")
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_feature_sets)

    return S; 

def stack_predicions(preds, S):
    n_vox= preds.shape[-1]
    stacked_pred = np.zeros((preds.shape[1], n_vox))
    for i in range(0, n_vox):
        z_test = np.array([preds[feature_j, :, i]  for feature_j in range(len(preds))])   
        stacked_pred[:, i] = np.dot(S[i, :], z_test)
    return stacked_pred;


###########################################################################
# Loading of MRI data

## Fmri data loading

# load pca for full brain reconstruction
fn="/scratch-scc/users/robert.scholz2/full_sample_pca91k.500comps.scaled.pkl"
pca = pk.load(open(fn,'rb'))
ess, ese = 5, 5
hrf_delay, stimulus_window = 3, 3
all_movie_sets = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]
movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05"]


# feature data loading
best_fsets = ["Llama-3.1-8B.4L3T1000W", "whisper-small.4L18T29S", "slow_r50.EMBD"]

feature_dicts = {}
for model_id in best_fsets:
    fn= f"{featred_dir}/actv-{model_id}.all_stimuli.s7ext.pca2000.npy"
    #if model_id=="slow_r50.LASTPOOL": fn = f"{featred_dir}/actv-slow_r50.LASTPOOL.all_stimuli.allred.pca2000.npy"
    all_features_dict = np.load(fn, allow_pickle=1).item();
    all_features_dict = {k.split("_")[-1]:v for k,v in all_features_dict.items()}
    print(model_id, len(all_features_dict.keys()), all_features_dict["s01e01a"].shape)
    feature_dicts[model_id] = all_features_dict




###########################################################################
# Variance partitioning


from itertools import combinations
def all_unordered_sublists(lst):
    return [list(c) for r in range(1, len(lst)+1) for c in combinations(lst, r)]

best_fsets = ["Llama-3.1-8B.4L3T1000W", "whisper-small.4L18T29S", "slow_r50.EMBD"]
print(best_fsets)

sublists = all_unordered_sublists(best_fsets)
for k in sublists: print(k)

n_vox = 500
score_fn = R2

# unique audio contribution in audio+vision

results = {}
for subject in [1,3,5]:
    print("#"*30)
    sub = f"sub-0{subject}"
    print(sub)
    fn=f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca.npy" 
    fmri = np.load(fn, allow_pickle=1).item()
    fmri={k.replace("_run-1",""): v for k, v in fmri.items() if not k.endswith("_run-2")};

    # Calculate prediction error on the "training set"
    training_errs = {}; preds_test ={}
    for model_id in best_fsets:
        print(model_id)
        train_err, y_optim_pred, y_optim, y_test, y_test_pred = train_and_predict(feature_dicts[model_id], fmri,\
            #model_id, sub, movies_train, ["figures"], ["figures"], use_prefit_model=True)
            model_id, sub, movies_train, ["friends-s06"], ["figures"], use_prefit_model=True)
        training_errs[model_id] = train_err
        preds_test[model_id] = y_test_pred
        
    train_data,test_data = y_optim, y_test
    err = np.stack([v for k,v in training_errs.items()])
    preds_test = np.stack([v for k,v in preds_test.items()])

    # full model; maybe not needed (?)
    # as its included in the later loop too...
    S= esimate_stacked_model_coeffs(err);
    stacked_pred_test = stack_predicions(preds_test, S);
    cortex_predic_all = pca.inverse_transform(stacked_pred_test[::10])
    cortex_target = pca.inverse_transform(test_data[::10])
    
    cortex_score_all= score_fn(cortex_predic_all[:, :59412], cortex_target[:, :59412])
    #combined_score[sub] = cortex_score_all
    print(cortex_score_all.min().round(2), cortex_score_all.max().round(2))
    print("---")

    subj_results={}
    for k, sublist in enumerate(sublists):
        print(sublist)
        mask = np.zeros(len(err), dtype=bool)
        for model_id in sublist: 
            mask[best_fsets.index(model_id)] = True
        print(mask, err[mask].shape)

        S= esimate_stacked_model_coeffs(err[mask]);
        stacked_pred_test = stack_predicions(preds_test[mask], S);
        cortex_predic_partial = pca.inverse_transform(stacked_pred_test[::10])
        cortex_score_partial  = score_fn(cortex_predic_partial[:, :59412], cortex_target[:, :59412])
        print(cortex_score_partial.min().round(2), cortex_score_partial.max().round(2))
        subj_results[tuple(sorted(sublist))] = cortex_score_partial

    results[sub] = subj_results


# transform it to be able to better handle later on:
res_raw = {
    key: np.stack([results[sub][key] for sub in results.keys()])
    for key in results["sub-01"].keys()
}

# keep it in the range of 0 and 1
res = {k:np.clip(v,0,1) for k,v in res_raw.items()}
for k in res.keys():
    print(res[k].shape, k)

# Save outputs
payload=dict(res_raw=res_raw, res=res, score_type="R2")
var_fn = res_dir+f"/varpart_all_stacked_on_s06_allsubjs_figures_ev10thTR.v2.R2.optim_on_friends06_and_score_on_figures.npy"
np.save(var_fn, payload)


############################################################################
# Variance partitioning

"""
res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"
var_fn = res_dir+f"/varpart_all_stacked_on_s06_allsubjs_figures_ev10thTR.v2.R2.optim_and_score_on_figures.npy"
p=np.load(var_fn, allow_pickle=1).item();
res_raw=p["res_raw"]; res=p["res"];""";

llm, aud, vis= [x for x in best_fsets]
# if we dont clip, the reconstruction error will be zero, but we get "negative contribution" by parts, which is harder to interpret.
#c = lambda x : np.clip(x,0,1) 
c = lambda x : x
def log(k): print(np.array([k.min(), k.max(), k.mean()]).round(2));

# full models
full = res[tuple(sorted(best_fsets))]
audi_vis = res[tuple(sorted([aud, vis]))]
audi_llm = res[tuple(sorted([aud, llm]))]
llm_vis = res[tuple(sorted([llm, vis]))]

# unique contributions to only the joint audio-visual / audi_llm
uniq_aud = audi_vis - res[tuple([vis])]
uniq_vis = audi_vis - res[tuple([aud])]
uniq_llm = audi_llm - res[tuple([aud])]

# unique contributions to the full three modality model
uniq_llm2 = full - c(audi_vis)
uniq_aud2 = full - c(llm_vis)
uniq_vis2 = full - c(audi_llm)

# unique contributions by two-modality combinations 
uniq_shared_audvis = full - res[tuple([llm])] - c(uniq_aud2) - c(uniq_vis2)
uniq_shared_audllm = full - res[tuple([vis])] - c(uniq_aud2) - c(uniq_llm2)
uniq_shared_llmvis = full - res[tuple([aud])] - c(uniq_vis2) - c(uniq_llm2)

# shared among all 
all_shared = full - c(uniq_shared_audvis) - c(uniq_shared_audllm) - c(uniq_shared_llmvis)
all_shared = all_shared - c(uniq_aud2) - c(uniq_vis2)  - c(uniq_llm2)

# total variance by modality
total_audio = full-uniq_llm2-uniq_vis2-uniq_shared_llmvis
total_vis = full-uniq_llm2-uniq_aud2-uniq_shared_audllm
total_llm = full-uniq_aud2-uniq_vis2-uniq_shared_audvis

recon = c(uniq_aud2) + c(uniq_vis2)  + c(uniq_llm2) + \
        c(uniq_shared_audvis) + c(uniq_shared_audllm) + c(uniq_shared_llmvis) +\
        all_shared 


res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"
var_fn = res_dir+f"/varpart_all_stacked_on_s06_allsubjs_figures_ev10thTR.v2.R2.optim_on_friends06_and_score_on_figures.npy"

p=np.load(var_fn, allow_pickle=1).item();
res_raw=p["res_raw"]; res=p["res"];

fn=f"{acc_dir}/retest_bw+pca_baseline.R2.npy"
figures_R2_baselines = np.array(np.load(fn, allow_pickle=1).item()["scores_bw"]["figures"])[:, :59412];
print(figures_R2_baselines.shape)


full = res[tuple(sorted(best_fsets))] # shape (3, 59412)

rfi = figures_R2_baselines-full

# uniq-audio or uniq-vis on full
percentage_unimodal = np.zeros_like(full.mean(0));
percentage_unimodal[:]=-1
mask= full.mean(0)>0.075
percentage_unimodal = (c(uniq_aud2)+c(uniq_vis2)).mean(0)/np.clip(full.mean(0),0.25**2,1)


res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"
var_deriv_fn = res_dir+f"/varpart_all_stacked_on_s06_allsubjs_figures_ev10thTR.v2.R2.optim_and_score_on_figures.deriv.npy"

p = dict(uniq_llm=uniq_llm, uniq_aud=uniq_aud, uniq_vis=uniq_vis, 
         uniq_llm2=uniq_llm2, uniq_aud2=uniq_aud2, uniq_vis2=uniq_vis2, 
         uniq_shared_audvis=uniq_shared_audvis, uniq_shared_audllm=uniq_shared_audllm, uniq_shared_llmvis=uniq_shared_llmvis, 
         all_shared=all_shared, percentage_unimodal=percentage_unimodal, recon=recon, rfi=rfi, full=full)
np.save(var_deriv_fn, p)
