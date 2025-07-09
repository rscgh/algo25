import os
root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"
featred_dir = f"{root_data_dir}/ann_brain_data/featred"
!hostname
print(os.getpid())

import numpy as np
import pickle as pk
from tqdm.auto import tqdm

from brainannlib.algonauts_funcs import load_stimulus_features
from brainannlib.algonauts_funcs import align_features_and_fmri_samples, train_sklearn_ridgecv
from brainannlib.algonauts_funcs import compute_encoding_accuracy

from brainannlib.stats_and_metrics import corr_score, R2, fast_corr#, fisher_z_to_r as zr, fisher_r_to_z as rz

from sklearn.model_selection import train_test_split
from brainannlib.visualization import plot_flatmap, plot_surface
from matplotlib import pyplot as plt

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


def setup_environment():
    """Set up environment"""
    import socket
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())
    

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



def run_stacked_regressions(score_fn = R2):
    # unique audio contribution in audio+vision
    print(best_fsets)
    
    combined_score = {}
    uniq_contrib = {}
    for subject in [1,3,5]:
        print("#"*30)
    
        sub = f"sub-0{subject}"
        fn=f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca.npy" 
        fmri = np.load(fn, allow_pickle=1).item()
        fmri={k.replace("_run-1",""): v for k, v in fmri.items() if not k.endswith("_run-2")};
    
        # Calculate prediction error on the "training set"
        training_errs = {}; preds_test ={}
        for model_id in best_fsets:
            print(model_id)
            train_err, y_optim_pred, y_optim, y_test, y_test_pred = train_and_predict(feature_dicts[model_id], fmri,\
                model_id, sub, movies_train, ["friends-s06"], ["figures"], use_prefit_model=True)
            training_errs[model_id] = train_err
            preds_test[model_id] = y_test_pred
            
        train_data,test_data = y_optim, y_test
        err = np.stack([v for k,v in training_errs.items()])
        preds_test = np.stack([v for k,v in preds_test.items()])
    
        S= esimate_stacked_model_coeffs(err);
        stacked_pred_test = stack_predicions(preds_test, S);
        cortex_predic_all = pca.inverse_transform(stacked_pred_test[::10])
        cortex_target = pca.inverse_transform(test_data[::10])
        
        cortex_score_all= score_fn(cortex_predic_all[:, :59412], cortex_target[:, :59412])
        combined_score[sub] = cortex_score_all
        print(cortex_score_all.min().round(2), cortex_score_all.max().round(2))
        print("---")
    
        uc={}
        for m, model_id in enumerate(best_fsets):
            print(model_id)
            mask = np.ones(len(err), dtype=bool)
            mask[m] = False
            print(mask, err[mask].shape)
    
            S= esimate_stacked_model_coeffs(err[mask]);
            stacked_pred_test = stack_predicions(preds_test[mask], S);
            cortex_predic_remove_mod = pca.inverse_transform(stacked_pred_test[::10])
            cortex_score_remove_mod= score_fn(cortex_predic_remove_mod[:, :59412], cortex_target[:, :59412])
            print(cortex_score_remove_mod.min().round(2), cortex_score_remove_mod.max().round(2))
            diff = np.clip(cortex_score_all, 0,1) - np.clip(cortex_score_remove_mod, 0,1);
            print(diff.min().round(2), diff.max().round(2))
            uc[model_id] = diff
    
        uniq_contrib[sub] = uc
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_comb", type=str, default="Llama-3.1-8B:4L7T1000W:3:1000",
                         help="Model definitions separated by ';' in format 'name:postfix'" \
                         "e.g. 'Llama-3.1-8B:4L7T1000W'")

    parser.add_argument("--stim_optim", type=str, default="...", help="Stimuli used for estimation of the stacking coefficients.")    
    parser.add_argument("--movies_test", type=str, default="", help="Stimuli used for scoring the stacked model on")    
    
    parser.add_argument("--subject_nrs", type=int, nargs="+", default=[1], help="Subject numbers")
    parser.add_argument("--comb_name", type=str, default="all_modalities", help="....'all'")
    args = parser.parse_args()

    root_data_dir = os.environ.get("ALGONAUTS_ROOT_DIR", "./data")
    device=setup_environment()

    all_embeddings=args.embedding_comb.split(";")














