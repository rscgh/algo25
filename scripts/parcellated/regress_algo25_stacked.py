

from brainannlib.algonauts_funcs import load_stimulus_features, load_fmri
from brainannlib.algonauts_funcs import align_features_and_fmri_samples, align_features_friends_s7_v2
from brainannlib.algonauts_funcs import compute_encoding_accuracy, plot_accuracy_on_brain
from brainannlib.algonauts_funcs import train_sklearn_ridgecv

import os, sys
import argparse
from tqdm.auto import tqdm
import pickle as pk
import numpy as np
import pandas as pd


# examplary: saves model predictions for the optim+test movie sets
#SAVE_MODEL_PREDS_DEBUG=True 
 

#!hostname

"""
Run as:
embeddings="whisper-small:4L18T29S:5:500"
embeddings="slowr50:loraft1:3:500;Llama-3.1-8B:4L7T1000W:3:500;algo_slowr50:None:5:100;whisper-small:4L18T29S:5:500"
combi="12345-5-6"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}"

cd /scratch-scc/users/robert.scholz2/emmy_code/algon25/scripts/parcellated
embeddings="whisper-small:4L18T29S:5:500;Llama-3.1-8B:4L7T1000W+OOD:3:500"
combi="12345-5-6"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}" --prep_submission

embeddings="whisper-small:4L18T29S:5:500"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}" --prep_submission --chaplin

embeddings="internvl-img-post-llm:x:x:x;whisper-small:4L18T29S:5:500;Llama-3.1-8B:4L7T1000W+OOD:3:500"

embeddings="ext_internvl-img-post-llm:x:0:0;whisper-small:4L18T29S:5:500;Llama-3.1-8B:4L7T1000W:3:500"

combi="12345-5-6"
embeddings="ext_internvl-img-post-llm:x:0:0;ext_internvl-post-text:x:0:0;ext_internvl-vit:x:0:0;whisper-small:4L18T29S:5:500;Llama-3.1-8B:4L7T1000W:3:500"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}" 

combi="12345bwf-15bwf-6"
embeddings="ext_internvl-img-post-llm:x:0:0;ext_internvl-post-text:x:0:0;ext_internvl-vit:x:0:0;whisper-small:4L18T29S:5:500;Llama-3.1-8B:4L7T1000W:3:500"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 2 3 5 --train_optim_test "${combi}" --prep_submission


--prep_submission


internvl-img-post-llm


list_of_embeddings=("algo_slowr50:None:5:100" "slow_r50:EMBD:5:100" "slow_r50:EMBDall:5:100" "slow_r50:EMBD15:5:100" "algo_slowr50:None:3:500" "slow_r50:EMBD:3:500" )
list_of_embeddings=("slow_r50:LASTPOOL:5:100"  "slow_r50:LASTPOOL:3:500")
list_of_embeddings=("slow_r50:EmbdPCAonAll:5:100"  "slow_r50:EmbdPCAonAll:3:500")
combi="12345-5-6"
for embeddings in "${list_of_embeddings[@]}"; do
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}"
done

list_of_embeddings=("InternVit:EMBD:5:100" "InternVit:EMBDs:5:100" "InternVit:EMBD:3:500" "InternVit:EMBDs:3:500")
for embeddings in "${list_of_embeddings[@]}"; do
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}"
done

# done
train_combis=("12345wf-12345wf-6" "12345-12345wf-6" "12345wf-12345-6" "12345-wf-6" ) # remove life+bourne
train_combis=("12345-5-6" "1234-5-6" "12345-4-6" "1235-4-6" "12345-45-6" "123-45-6" "12345-12345-6")

# todo
train_combis=("12345-5-6" "1234-5-6" "12345-4-6" "1235-4-6" "12345-45-6" "123-45-6" "12345-12345-6")
train_combis=("12345-bwfl-6" "12345bwfl-12345bwfl-6" "12345-12345bwfl-6" "12345bwfl-12345-6")


train_combis=("12345-bwfl-6" "12345bwfl-12345bwfl-6" "12345-12345bwfl-6" "12345bwfl-12345-6")
train_combis=("12345-bwf-6" "12345bwf-12345bwf-6" "12345-12345bwf-6" "12345bwf-12345-6") # remove life
for combi in "${train_combis[@]}"; do
    python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}"
done

train_combis=("12345-5-f" "12345-45-f" "12345w-12345-f" "12345-w-f" "12345-w6-f" "12345wf-12345wf-f" "12345-bw-6" "12345bw-bw-f")
for combi in "${train_combis[@]}"; do
    python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}"
done

train_combis=("f-f-f" "12345f-f-f" "6-6-6" "123456-6-6") # max overfitting



embeddings="llama-3-8b-bnb-4bit-contpretr-lora-friends-0.1:4L7T1000W:3:500;algo_slowr50:None:5:100;whisper-small:4L18T29S:5:500"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "12345-5-6"

embeddings="llama-3-8b-bnb-4bit-contpretr-lora-friends-0.1:4L7T1000W:3:500;algo_slowr50:None:5:100;whisper-small:4L18T29S:5:500;AST:4L6T10S:3:500;Qwen2.5-7B:4L3T1000W:3:500;whisper-large-v3:4L18T29S:3:500;SmolLM2-1.7B:4L3T1000W:3:500"
python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "12345-5-6"

train_combis=("12345wfbl-12345wfbl-6" "12345wfbl-12345wfbl-f"  "12345wfbl-12345wfb-6" "12345wfbl-12345wfb-f")

python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "12345wfbl-12345wfbl-6"

train_combis=("12345-5-f" "12345-45-f" "12345w-12345-f" "12345-w-f" "12345-w6-f" "12345wf-12345wf-f" "12345-bw-6" "12345bw-bw-f")
for combi in "${train_combis[@]}"; do
    python -u regress_algo25_stacked.py --embedding_comb "${embeddings}" --subject_nrs 1 --train_optim_test "${combi}"
done


""";


root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
featred_dir = f"{root_data_dir}/ann_brain_data/featred"
model_dir =f"{root_data_dir}/ann_brain_data/models"
pred_dir = f"{root_data_dir}/ann_brain_data/preds"


def align_features_and_fmri_samples_v2(features, fmri, excluded_samples_start, excluded_samples_end,\
                                    hrf_delay, stimulus_window, stim_sets, v=False, n_targets=1000, custom_episodes=[]):
    # initialize empty array with 1000 parcels
    # this will return a matrix of concatenated timepoints across episodes/movies x 1000 parcels
    aligned_fmri = np.empty((0, n_targets), dtype=np.float32) 
    # the feature matrix will have the same # of timepoints as the aligned_fmri matrix
    aligned_features = []

    isfirst = True; # just for verbose output/debugging

    for stimset in stim_sets:
        # stimset e.g. "s01", "s02", ... "bourne", "life", ...
        # but also acommodating their style: "friends-s01", "movie10-bourne" ...
        if stimset == "custom_episodes":
            episodes_in_set = custom_episodes;
        else:
            stimset = stimset.split("-")[-1]
            episodes_in_set = [key for key in fmri if key.startswith(stimset)]

        if v>=1: print(stimset, len(episodes_in_set), episodes_in_set[:3])

        for episode in episodes_in_set:
            fmri_run = fmri[episode][excluded_samples_start:-excluded_samples_end]
            n_trs = len(fmri_run)
            #print(n_trs, aligned_fmri.shape, fmri_run.shape )
            aligned_fmri = np.append(aligned_fmri, fmri_run, 0)

            if isfirst and v>=2: print("\nsplit", episode, fmri_run.shape)
                
            # for each tr in the fmri-data
            for s in range(n_trs):
                # create an empty feature vector, to which the feautures 
                # from each modality will be concatenated
                # in the end f_all is appended to aligned_features
                f_all = np.empty(0)

                # for each modalitly [i.e. visual, audio or language]
                for mod, mod_features in features.items():
                    
                    episode_feat_len= len(mod_features[episode])
                    stim_wind = stimulus_window if not(isinstance(stimulus_window, dict)) else stimulus_window[mod]

                    idx_start = max(excluded_samples_start, s + excluded_samples_start - hrf_delay - stim_wind + 1)
                    idx_end = idx_start + stim_wind
                    if idx_end > episode_feat_len:
                        idx_end = episode_feat_len
                        idx_start = idx_end - stim_wind

                    # we get the feature vectors for multiple samples (N=stimulus_window), and flatten them
                    f = mod_features[episode][idx_start:idx_end].flatten()
                    
                    if isfirst and (s in [0, 1, 2, 10, 30, n_trs-2, n_trs-1]) and v>=2:
                        print(mod, f"s={s}, idx_start={idx_start}, idx_end={idx_end}, idx_max={episode_feat_len}, f={f.shape}, {f.flatten().shape}")

                    f_all = np.append(f_all, f)

                if isfirst and (s in [0, 1, 2, 10, 30, n_trs-2, n_trs-1]) and v>=2:
                    print(f"s={s}, f_all_shape={f_all.shape}")
                        
                aligned_features.append(f_all)
            
            isfirst=False;

    return np.array(aligned_features, dtype=np.float32), aligned_fmri



def setup_environment():
    """Set up environment"""
    import socket
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())



    
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import os;
from brainannlib.algonauts_funcs import train_sklearn_ridgecv

class StackedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, sub="sub-XX"):

        self.models_ = {}
        self.model_ids = []
        self.stacked_coeffs = None;
        self.save_indv_models = True
        self.model_root_path = model_dir; # {algoanuts_root}/ann_brain_data/models/
        self.sub=sub

    def fit_indiv_model_(self, model_id, x_train, y_train, train_type="n500.sw3.12345"):
        # model_id e.g. "whisper-small.4L18T29S.5.500.12345"
        model_fn = f"{self.model_root_path}/full_brain_models.{model_id}.{self.sub}.singlemodel.pkl"
        #model_fn = f"{model_dir}/full_brain_models.{feature_set}.{args.training_src}.sub-0{subject}.{stimulus_window}.singlemodel.pkl"
        if os.path.exists(model_fn):
            print("load existing model", model_fn)
            import pickle as pk
            model = pk.load(open(model_fn,'rb'))
        else:
            # fit the model actually
            print("Training model:", model_id, "on", x_train.shape, "->", y_train.shape)
            model = train_sklearn_ridgecv(x_train, y_train)
            if self.save_indv_models:
                print("Saving", model_fn)
                import pickle as pk
                with open(model_fn, 'wb') as f:
                   pk.dump(model, f)

        if not model_id in self.model_ids:
            self.model_ids.append(model_id) # keep track of the order
        self.models_[model_id] = model

    def fit_indiv_models(self, x_train, y_train):
        for model_id, xt in x_train.items():
            self.fit_indiv_model_(model_id, xt, y_train)

    def indiv_model_predict(self, model_id, x_data):
        if not(model_id in self.models_.keys()):
            raise ValueError(f"Indiv. Model {model_id} does not yet exist, please call fit_indiv_model_ to fit it first.")
        
        indiv_model = self.models_[model_id]
        return indiv_model.predict(x_data)
    
    def esimate_stacked_model_coeffs(self, err):
        # intialize a few matrices

        if isinstance(err, dict):
            err = np.stack([err[mid] for mid in self.model_ids])

        n_feature_sets=len(err)
        n_vox = err[0].shape[-1]

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
        
        self.stacked_coeffs = S;
    
    def fit_stacked_(self, x_optim, y_optim, y_optim_pred={}):
        errs = {} 
        #if isinstance(x_optim, dict): x_optim = [x_optim[model_id] for model_id in self.model_ids]
        # get individual predictions 
        
        # if instead of prev fitting individual models, we just use
        # predictions directly, make sure to keep track of which model
        # preditions were used in self.model_ids 
        for key in y_optim_pred.keys():
            if not(key in self.model_ids): self.model_ids.append(key)

        for i, model_id in enumerate(self.model_ids):
            y_pred = None
            if model_id in y_optim_pred.keys():
                # use directly if given
                y_pred=y_optim_pred[model_id] 
            else:
                # otherwise use intidividual model to generate them
                y_pred = self.indiv_model_predict(model_id, x_optim[model_id])
            errs[model_id]= y_optim - y_pred
        
        # estimate stacked model coeffs
        self.esimate_stacked_model_coeffs(errs);

    def predict(self, x_pred, y_pred = {}):
        # x_pred: model_id -> features
        # y_pred: preplaces the internal calculation of individual predicitons
        preds = []
        for mid in self.model_ids:
            pred = y_pred[mid] if mid in y_pred.keys() \
                else self.indiv_model_predict(mid, x_pred[mid])
            preds.append(pred);
        preds = np.stack(preds)

        n_vox= preds.shape[-1]
        stacked_pred = np.zeros((preds.shape[1], n_vox))
        for i in range(0, n_vox):
            z_test = np.array([preds[feature_j, :, i]  for feature_j in range(len(preds))])   
            stacked_pred[:, i] = np.dot(self.stacked_coeffs[i, :], z_test)
        return stacked_pred;


def compile_features_for_def(model_name, postfix, n_feat):
    fn= os.path.join(featred_dir, f"actv-{model_name}.{postfix}.all_stimuli.s7ext.pca2000.npy");
    mod_features=None
    if not(os.path.exists(fn)):
        if model_name=="algo_slowr50":
            print("loading (hardcoded):")
            mod_features = load_stimulus_features(root_data_dir, ["visual"])['visual']
            mod_features = {k:v[:,:n_feat] for k,v in mod_features.items()}
            print(model_name, postfix, mod_features["s01e01a"].shape)
        else:
            print("Error: couldnt find", fn)
            #print("So ignoring:", model_def)
    else:
        #print("loading:", model_def)
        print("Loading:", fn)
        mod_features = np.load(fn, allow_pickle=True).item()
        mod_features = {k.split("_")[-1]: v for k,v in mod_features.items()}
        mod_features = {k:v[:,:n_feat] for k,v in mod_features.items()}
        print(model_name, mod_features["s01e01a"].shape)
    
    return mod_features

# untested
def transform_flat_to_dict_preds(preds_flat, fmri, movies_sets, ess=5, ese=5):
    # transforms pred_flat (n_trs, n_vox) into a dict[epi] = (n_tr_epi, n_vox)
    # this mirrors the standard align_features_and_fmri_samples
    # (for easier saving)
    preds = {}
    start = 0
    for stimset in movies_sets:
        # stimset e.g. "s01", "s02", ... "bourne", "life", ...
        # but also acommodating the long style: "friends-s01", "movie10-bourne" ...
        stimset = stimset.split("-")[-1] # e.g. 
        episodes_in_set = [key for key in fmri if key.startswith(stimset)]

        for key in episodes_in_set:
            fmri_run = fmri[key][ess:-ese]
            n_trs = len(fmri_run)
            preds[key] = preds_flat[start:start+n_trs]
            start = start+n_trs;
    
    return preds; 

# untested
def load_preds_and_align_mri(preds_dict, fmri, movie_sets, ess=5, ese=5, custom_episodes=[]):
    # fmri is a dict[epi]=(n_tr_epi, n_vox)
    preds_flat = []
    mri_flat=[]
    for stimset in movie_sets:
        # stimset e.g. "s01", "s02", ... "bourne", "life", ...
        # but also acommodating the long style: "friends-s01", "movie10-bourne" ...
        stimset = stimset.split("-")[-1] # e.g. 
        episodes_in_set = [key for key in fmri if key.startswith(stimset)]
        if stimset == "custom_episodes": 
            episodes_in_set=custom_episodes;
        for key in episodes_in_set:
            preds_flat += [preds_dict[key]]
            mri_flat += [fmri[key][ess:-ese]]
    
    preds_flat=np.concatenate(preds_flat, axis=0)
    mri_flat=np.concatenate(mri_flat, axis=0)
    return preds_flat, mri_flat; 


def fit_and_score_stacked_regression(fmri, movies_train, movies_optim, movies_test, subject, args):
    hrf_delay = 3; excluded_samples_start, excluded_samples_end = 5, 5

    print("#"*30)
    sub = f"sub-0{subject}"
    print(sub)

    model = StackedRegressor(sub);
    y_optim_pred, y_test_pred = {}, {}
    y_optim, y_test = None, None;
    
    for model_def in args.embedding_comb.split(";"):
        # model_def e.g. "llama-3b:10030:3:1000"
        print("model_def:", model_def);
        model_name, postfix, stim_window, n_feat = model_def.split(":")
        #model_id = model_name+"."+postfix
        # here e.g. "whisper-small.4L18T29S.5.500.12345"
        training = args.train_optim_test.split("-")[0]
        model_id = f"{model_name}.{postfix}.{stim_window}.{n_feat}.{training}"
        if model_name.startswith("ext_"):
            # make it easier to use external files
            model_id = model_name[4:]
            print("Using external model preds:", model_id)

        #if postfix=="kunalpca250":
        #    print("Set HRF-delay to zero")
        #    hrf_delay=0;

        stim_window = int(stim_window); n_feat=int(n_feat);

        # if not optimization movies given, just use the same as training movies
        # could lead to overfitting
        if movies_optim is None: 
            movies_optim = movies_train;

        ## if model_id has already predictions saved, 
        ## just load them depending on movies_optim & movies_test
        pred_fn = f"{pred_dir}/parcel_brain_models.{model_id}.{sub}.all_predictions.npy"
        print(pred_fn)
        
        if os.path.exists(pred_fn):
            print("Load existing predictions:", pred_fn)
            #preds_dict = {"s01e02b":"mri_data"}
            preds_dict = np.load(pred_fn, allow_pickle=1).item()
            if model_name.startswith("ext_"):
                preds_dict = {k: v[5:-5] for k, v in preds_dict.items()}

            y_optim_mid_pred, y_optim = load_preds_and_align_mri(preds_dict, fmri, movies_optim, 
                           ess=excluded_samples_start, ese=excluded_samples_end)
            y_test_mid_pred, y_test = load_preds_and_align_mri(preds_dict, fmri, movies_test, 
                            ess=excluded_samples_start, ese=excluded_samples_end)
            
            y_optim_pred[model_id] = y_optim_mid_pred
            y_test_pred[model_id]= y_test_mid_pred
            continue; 

        # otherwise load features
        # fit model (or load if existing)
        # and then predict
        enc_features = compile_features_for_def(model_name, postfix, n_feat)
        enc_features = {"visual": enc_features}
        
        x_train, y_train = align_features_and_fmri_samples_v2(enc_features, fmri, excluded_samples_start, 
            excluded_samples_end, hrf_delay, stim_window, movies_train)

        # check first if the predictions exist somewhere and can be loaded from a dict
        # for indiviudal models, and only fit the model if needed.
        model.fit_indiv_model_(model_id, x_train, y_train)
        del x_train, y_train;
        
        x_optim_mid, y_optim = align_features_and_fmri_samples_v2(enc_features, fmri, excluded_samples_start, 
            excluded_samples_end, hrf_delay, stim_window, movies_optim)
        y_optim_pred[model_id] = model.indiv_model_predict(model_id, x_optim_mid)

        x_test_mid, y_test = align_features_and_fmri_samples_v2(enc_features, fmri, excluded_samples_start, 
            excluded_samples_end, hrf_delay, stim_window, movies_test)
        y_test_pred[model_id] = model.indiv_model_predict(model_id, x_test_mid)

        if args.save_predictions:
            all_preds_dict = {} # all optim and test
            optim_dict = transform_flat_to_dict_preds(y_optim_pred[model_id], fmri, movies_optim, ess=5, ese=5)
            test_dict = transform_flat_to_dict_preds(y_test_pred[model_id], fmri, movies_test, ess=5, ese=5)
            all_preds_dict.update(optim_dict)
            all_preds_dict.update(test_dict)
            print(f"Saving {len(all_preds_dict.keys())} episode predictions to:", pred_fn)
            np.save(pred_fn, all_preds_dict)

            y_test_mid_pred2, y_test2 = load_preds_and_align_mri(all_preds_dict, fmri, movies_test, 
                    ess=excluded_samples_start, ese=excluded_samples_end)
            
            print(y_test_pred[model_id].shape, y_test_mid_pred2.shape)
            print(y_test.shape, y_test2.shape)
            print(np.allclose(y_test_pred[model_id], y_test_mid_pred2))
            print(np.allclose(y_test, y_test2))
            del test_dict, optim_dict, all_preds_dict


    print(y_optim.shape)
    for k,v in y_optim_pred.items(): print(k,v.shape)
    print("Fitting stacked model.");
    model.fit_stacked_(None, y_optim, y_optim_pred=y_optim_pred)
    
    print("Predicting.");
    stacked_pred = model.predict(None, y_pred = y_test_pred)

    encoding_accuracy, mean_acc = compute_encoding_accuracy(y_test, stacked_pred, subject, "all")
    print(args.embedding_comb)
    print(mean_acc.round(3), "\t", encoding_accuracy.min().round(3), encoding_accuracy.max().round(3))
    print("---")

    score = [args.train_optim_test, "_".join(movies_test), (0), mean_acc, encoding_accuracy]
    return model, score, stacked_pred

########################################################################
# OOD submission


def align_features_and_fmri_samples_ood_v2(features_ood, root_data_dir, sub= 1, hrf_delay = 3, stimulus_window = 5, episodes=None):
    aligned_features_ood = {}
    
    ### Load the OOD movie fMRI samples ###
    samples_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{sub}', 'target_sample_number',
        f'sub-0{sub}_ood_fmri_samples.npy')
    fmri_samples = np.load(samples_dir, allow_pickle=True).item()
    #print("Availble fMRI:", fmri_samples.keys())
    if episodes is None: episodes = fmri_samples.keys()
    #print("Using:", episodes)
    #fmri_samples={k,v for k,v in fmri_samples.items() if k in episodes}

    ### Loop over the OOD movies ###
    for epi, samples in fmri_samples.items():
        if not epi in episodes: continue
        features_epi = []
        ### Loop over fMRI samples ###
        for s in range(samples):
            f_all = np.empty(0)

            for mod in features_ood.keys():

                if s < (stimulus_window + hrf_delay):
                    idx_start = 0
                    idx_end = idx_start + stimulus_window
                else:
                    idx_start = s - hrf_delay - stimulus_window + 1
                    idx_end = idx_start + stimulus_window
                # In case there are less visual/audio feature samples
                # than fMRI samples minus the hrf_delay, use the last N
                # visual/audio feature samples available (where N is
                # defined by the 'stimulus_window' variable)
                if idx_end > len(features_ood[mod][epi]):
                    idx_end = len(features_ood[mod][epi])
                    idx_start = idx_end - stimulus_window
                f = features_ood[mod][epi][idx_start:idx_end]
                f_all = np.append(f_all, f.flatten())
                #if epi not in ['chaplin1', 'chaplin2']:
            ### Append the stimulus features of all modalities for this sample ###
            features_epi.append(f_all)

        ### Add the episode stimulus features to the features dictionary ###
        aligned_features_ood[epi] = np.asarray(features_epi, dtype=np.float32)

    return aligned_features_ood



def only_predict_stacked_regression(model, movies_test, subject, args):

    sub = f"sub-0{subject}"
    y_test_pred = {c:{} for c in movies_test}

    for model_def in args.embedding_comb.split(";"):
        # model_def e.g. "llama-3b:10030:3:1000"
        print("model_def:", model_def);
        model_name, postfix, stim_window, n_feat = model_def.split(":")
        #model_id = model_name+"."+postfix
        # here e.g. "whisper-small.4L18T29S.5.500.12345"
        training = args.train_optim_test.split("-")[0]
        model_id = f"{model_name}.{postfix}.{stim_window}.{n_feat}.{training}"
        if model_name.startswith("ext_"):
            model_id = model_name[4:]
            print("Using external model preds:", model_id)
        
        stim_window = int(stim_window); n_feat=int(n_feat);

        ## if model_id has already predictions saved, 
        ## just load them depending on movies_optim & movies_test
        pred_fn = f"{pred_dir}/parcel_brain_models.{model_id}.{sub}.all_predictions.npy"
        #print(pred_fn)
        if os.path.exists(pred_fn):
            print("Load existing predictions:", pred_fn)
            #preds_dict = {"s01e02b":"mri_data"}
            preds_dict = np.load(pred_fn, allow_pickle=1).item()
            #if model_name.startswith("ext_") and not("slowr50klora" in model_name):
            #    preds_dict = {k: v[5:-5] for k, v in preds_dict.items()}
            #if is_episode:
            #y_test_mid_pred, y_test = load_preds_and_align_mri(preds_dict, fmri, ["custom_episodes"], 
            #                ess=excluded_samples_start, ese=excluded_samples_end, custom_episodes=movies_test)
            if np.all([c in preds_dict.keys() for c in movies_test]):
                print("adding")
                for clip in movies_test:
                    y_test_pred[clip][model_id] = preds_dict[clip]
                    #print(model_id, clip, preds_dict[clip].shape)
                continue; 
            
            print("Not all predictions present in the loaded file, missing")
            print([c for c in movies_test if not c in preds_dict.keys()])

        # otherwise load features
        # fit model (or load if existing)
        # and then predict
        enc_features = compile_features_for_def(model_name, postfix, n_feat)
        enc_features = {"visual": enc_features}

        x_test_mid = align_features_and_fmri_samples_ood_v2(enc_features, root_data_dir, sub=subject, stimulus_window=stim_window, episodes=movies_test)

        for clip in x_test_mid.keys():
            #print(model_id, clip,x_test_mid[clip].shape)
            y_test_pred[clip][model_id] = model.indiv_model_predict(model_id, x_test_mid[clip])
            #print(model_id, clip, y_test_pred[clip][model_id].shape)

        
    print("Predicting.");
    stacked_pred={}
    for clip in movies_test:
        stacked_pred[clip] = model.predict(None, y_pred = y_test_pred[clip]).astype(np.float32)
        print(clip, stacked_pred[clip].shape)

    
    return stacked_pred

########################################################################


def get_train_optim_test_movies(train_optim_test="123456-bwfl-f"):
    #all_movie_sets = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", \
    #              "friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]
    # ood-wot,pulpfiction,planetearth,passepartout,mononoke
    train, optim, test = train_optim_test.split("-");
    datamap = {n:f"friends-s0{n}" for n in "123456"}
    datamap.update({ name.split("-")[-1][:1]: name for name in ["movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]})
    
    movies_train = [datamap[x] for x in train]
    movies_optim = [datamap[x] for x in optim]
    movies_test = [datamap[x] for x in test]
    
    return movies_train, movies_optim, movies_test 
    
########################################################################





if __name__ == "__main__":
    print(os.getpid())

    #root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
    #actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    #featred_dir = f"{root_data_dir}/ann_brain_data/featred"


    parser = argparse.ArgumentParser(description="Process DNN activation features.")
    parser.add_argument("--embedding_comb", type=str, default="Llama-3.1-8B:4L7T1000W:3:1000",
                         help="Model definitions separated by ';' in format 'name:postfix:stim_window:n_features'" \
                         "e.g. 'Llama-3.1-8B:4L7T1000W:3:1000'" \
                         "Defines: model_name, postfix, stim_window, and n_features" \
                         "loads per-episode ANN embeddings dict based on model_name and postfix from featred_dir" \
                         "then aligns relevant episodes with mri based on stim_window, and retains only n_features")
    parser.add_argument("--subject_nrs", type=int, nargs="+", default=[1], help="Subject numbers")
    parser.add_argument("--train_optim_test", type=str, default="123456-bwl-f", help=\
                        """defines the datasets used in order train-optim-test, 
                        whereby train is used for per ANN ridge regressions 
                        (unused if predictions already exist and placed in "pred_dir" with the right filename) 
                        optim is used for estimating the stacking weights
                        and test is used to evaluate the stacked model on """)
    #parser.add_argument("--comb_name", type=str, default="bestForSure", help="....'all'")
    #parser.add_argument("--target", type=str, default="1000parcels", help="1000parcels or 500networks")
    #parser.add_argument("--save", action="store_true", help="Save the estimated models")
    parser.add_argument("--chaplin", action="store_true", help="Save the predctions")
    parser.add_argument("--save_predictions", action="store_true", help="Save the predctions")
    parser.add_argument("--save_stack_test_predictions", action="store_true", help="Save the predctions")
    parser.add_argument("--prep_submission", action="store_true", help="")

    parser.add_argument("--comb_name", type=str, default="llama+whisper_default")


    args = parser.parse_args()

    print("\n\n" + "----" * 20)
    setup_environment()
    print("Arguments given:", args)

    all_ood_clips = ["mononoke1", "mononoke2",  "passepartout1", "passepartout2", "planetearth1", "planetearth2", "pulpfiction1", "pulpfiction2", "wot1", "wot2", "chaplin1", "chaplin2"]
    
    movies_train, movies_optim, movies_test = get_train_optim_test_movies(args.train_optim_test);
    submission_predictions_ood={}

    for subject in args.subject_nrs:
        fmri=None;
        fmri = load_fmri(root_data_dir, subject) # subject = 1; 
        model, score, stacked_pred= fit_and_score_stacked_regression(fmri, movies_train, movies_optim, movies_test, subject, args)

        if args.save_stack_test_predictions:
            fn= f"{pred_dir}/model_comb_{args.comb_name}_{args.train_optim_test}_predictions.npy"
            np.save(fn, stacked_pred)
            print("Saved:", fn)

        comb_embd_results_fn = "algo25_combined_embds_stacked_results.csv"

        cols=["model_combination", "sub", "train", "test", "train_feat_shape", "mean_acc"] + list(np.arange(1000))
        df = pd.read_csv(comb_embd_results_fn) if os.path.exists(comb_embd_results_fn) else pd.DataFrame(columns=cols)
        print("Combined embeddings df of len", len(df))
        
        id_cols = cols[:4]  # Columns to check as identifier
        nc = len(id_cols);
        
        for res in [score]:
            train_optim_test, test, train_feat_shape, mean_acc, encoding_accuracy = res
            new_row= [args.embedding_comb] + [subject] + [train_optim_test, test, train_feat_shape, mean_acc.round(4)] + list(encoding_accuracy);
            idx = df.loc[(df[id_cols] == new_row[:nc]).all(axis=1)].index
            
            print(idx, new_row[:len(new_row)-1000])
            if len(idx)>0: #df.loc[idx] = new_row # replace existing row
                df = df.drop(idx[0])
            #else: 
            df.loc[len(df)] = new_row # add new row

        print("Saving to:", comb_embd_results_fn)
        df.to_csv(comb_embd_results_fn, index=False)
    
        if args.prep_submission:

            ood_clips = all_ood_clips[:10] # w/o chaplin
            if args.chaplin: 
                ood_clips=all_ood_clips[10:] # only chaplin
            
            per_clip_predictions = only_predict_stacked_regression(model, ood_clips, subject, args)
            submission_predictions_ood[f"sub-0{subject}"] = per_clip_predictions
    

    ## done unless we are preparing a submission
    if args.prep_submission:

        # Save the predicted fMRI dictionary as a .npy file
        output_file = f"{pred_dir}/model_comb_{args.comb_name}_trained_on_{args.train_optim_test}-ood_predictions.npy"
        if args.chaplin: 
            assert os.path.exists(output_file), f"File needs to exist, but doesnt: {output_file}"
            ood_preds = np.load(output_file, allow_pickle=1).item()
            for sub in submission_predictions_ood.keys():
                ood_preds[sub].update(submission_predictions_ood[sub])
            submission_predictions_ood = ood_preds

        np.save(output_file, submission_predictions_ood)
        print(f"Formatted predictions saved to: {output_file}")

        if args.chaplin: 
            
            for sub in submission_predictions_ood.keys():
                print(sub, submission_predictions_ood[sub].keys())

            subject = list(submission_predictions_ood.keys())[0]
            assert np.all([x in submission_predictions_ood[subject].keys() for x in all_ood_clips])
            # Zip the saved file for submission
            zip_file = f"{pred_dir}/model_comb_{args.comb_name}_trained_on_{args.train_optim_test}-ood_predictions.zip"
            import zipfile
            with zipfile.ZipFile(zip_file, 'w') as zipf:
                zipf.write(output_file, os.path.basename(output_file))
            print(f"Submission file successfully zipped as: {zip_file}")
            #os.remove(output_file);