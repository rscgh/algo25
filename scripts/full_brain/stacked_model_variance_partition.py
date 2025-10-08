import os
import numpy as np
import pickle as pk

from brainannlib.stats_and_metrics import R2
from brainannlib.algonauts_funcs import (
    align_features_and_fmri_samples, 
    train_sklearn_ridgecv
)

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
featred_dir = f"{root_data_dir}/ann_brain_data/featred"
res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"

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
