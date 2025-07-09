import os
import argparse
import numpy as np
import pickle as pk
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from brainannlib.algonauts_funcs import (
    align_features_and_fmri_samples,
    train_sklearn_ridgecv,
    compute_encoding_accuracy,
)
from brainannlib.stats_and_metrics import corr_score, compute_block_perm_indices, fast_corr
from brainannlib.feature_loading import quickalign, load_and_concat_activations
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

STIM_WINDOW_IN_TRs = 3;
HRF_DELAY_MIN_IN_TRs = 3;

def setup_environment():
    """Set up environment"""
    import socket
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())


def estimate_models(root_data_dir, feature_set, subject_nrs, enc_features, all_training_keys):
    print("\n\n" + "----" * 20)
    print("Model estimation (5-fold-CV)")

    np.random.seed(hash(feature_set) % 2**32)
    np.random.shuffle(all_training_keys)

    enc_features = {"visual": enc_features}


    for subject in subject_nrs:
            sub = f"sub-0{subject}"
            print("Training model for:", sub)
            fmri_fn = f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca.npy"
            #fmri_fn = f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca2k.npy"
            fmri = np.load(fmri_fn, allow_pickle=True).item()
            fmri = {k.replace("_run-1", ""): v for k, v in fmri.items() if not k.endswith("_run-2")}
            print(f"frmi for subj {sub} has {len(fmri.keys())} episodes");

            ########################################
            existing_training_keys=[epi for epi in all_training_keys if epi in fmri.keys()]
            if len(existing_training_keys) < len( all_training_keys):
                missing= [epi for epi in all_training_keys if not(epi in fmri.keys())]
                print("missing fmri-data for episodes:", missing)
            all_training_keys =np.array(existing_training_keys);
            ########################################

            print("Folds:")
            
            kf = KFold(n_splits=5)
            for fold, (train, test) in tqdm(enumerate(kf.split(all_training_keys)), total=5):

                model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{feature_set}.s1to5.{sub}.f{fold}.pkl"
                if os.path.exists(model_fn):
                    print(fold, "skipped. Aleady exists:", model_fn)
                    continue;

                features_train, fmri_train = align_features_and_fmri_samples(
                    enc_features, fmri, 5, 5, HRF_DELAY_MIN_IN_TRs, STIM_WINDOW_IN_TRs, ["custom_episodes"], 
                    n_targets=fmri[list(fmri.keys())[0]].shape[1],
                    custom_episodes=all_training_keys[train]
                )
                print(fold, "\t", features_train.shape, fmri_train.shape, end=" ... ")
                model = train_sklearn_ridgecv(features_train, fmri_train)

                pk.dump(model, open(model_fn, "wb"))
                print(f"saved as :", model_fn)


def generate_average_models(root_data_dir, feature_set, subject_nrs):
    print("\n\n" + "----" * 20)
    print("Generating average models for each subject")

    for subject in subject_nrs:
        sub = f"sub-0{subject}"
        c, i = [], []

        for fold in range(5):
            model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{feature_set}.s1to5.{sub}.f{fold}.pkl"
            model = pk.load(open(model_fn, "rb"))
            c.append(model.coef_)
            i.append(model.intercept_)

        model.coef_, model.intercept_ = np.mean(c, axis=0), np.mean(i, axis=0)
        avg_model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{feature_set}.s1to5.{sub}.avgmodel.pkl"
        pk.dump(model, open(avg_model_fn, "wb"))
        print("saved:", avg_model_fn)


def score_average_model(root_data_dir, feature_set, subject_nrs, enc_features, all_movie_sets, train_movies):
    print("\n\n" + "----" * 20)
    print("Scoring average model")

    #test_movies = ["friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]
    #pca_path = "/scratch-scc/users/robert.scholz2/full_sample_pca91k.2000comps.scaled.pkl"
    pca_path = "/scratch-scc/users/robert.scholz2/full_sample_pca91k.500comps.scaled.pkl"
    pca = pk.load(open(pca_path,'rb'))

    enc_features = {"visual": enc_features}

    for subject in subject_nrs:
        sub = f"sub-0{subject}"

        score_fn = f"{root_data_dir}/ann_brain_data/scores/full_brain_models.{feature_set}.s1to5.{sub}.scores.avgmodel.npy"
        if os.path.exists(score_fn):
            print(f"Skipping {sub} as exists already:", score_fn)
            #print(f"Overwriting {sub} as exists already:", score_fn)
            continue;

        #fmri_fn = f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca2k.npy"
        fmri_fn = f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca.npy"
        fmri = np.load(fmri_fn, allow_pickle=True).item()
        fmri = {k.replace("_run-1", ""): v for k, v in fmri.items() if not k.endswith("_run-2")}

        # resulting keys: 'life05', 's01e01a', 's01e01b', 

        # **Load the saved average model**
        avg_model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{feature_set}.s1to5.{sub}.avgmodel.pkl"
        if not os.path.exists(avg_model_fn):
            print(f"Skipping {sub} - Model not found: {avg_model_fn}")
            continue

        model = pk.load(open(avg_model_fn, "rb"))
        print(f"// {sub} - {feature_set}")
        scores_bw_mvset, scores_pc_mvset = {}, {}

        for vstim in tqdm(all_movie_sets, total=len(all_movie_sets)):
            features_val, fmri_val = align_features_and_fmri_samples(
                enc_features, fmri, 5, 5, HRF_DELAY_MIN_IN_TRs, STIM_WINDOW_IN_TRs,
                [vstim], n_targets=fmri[list(fmri.keys())[0]].shape[1])
            fmri_val_pred = model.predict(features_val)

            cortex_target = pca.inverse_transform(fmri_val)
            cortex_predic = pca.inverse_transform(fmri_val_pred)

            with np.errstate(invalid="ignore"):
                bwscore = corr_score(cortex_target, cortex_predic)

            encoding_accuracy, _ = compute_encoding_accuracy(fmri_val, fmri_val_pred)
            scores_bw_mvset[vstim] = bwscore
            scores_pc_mvset[vstim] = encoding_accuracy

            is_test = vstim not in train_movies
            print(f"{vstim:<15} {bwscore.mean():.3f}  {'test' if is_test else 'train'}")

        print("Contine with scoring the episodes:")

        scores_bw_epi ={}
        scores_pc_epi ={}
        indiv_stimuli = [k for k in fmri.keys() if not(k.startswith("s0"))]
        indiv_stimuli = indiv_stimuli + [k for k in fmri.keys() if k.startswith("s06")]

        for test_episode in tqdm(indiv_stimuli, desc="scoring episodes for "+sub):
            features_val, fmri_val = align_features_and_fmri_samples(enc_features, fmri, 
                    5, 5, HRF_DELAY_MIN_IN_TRs, STIM_WINDOW_IN_TRs, \
                    ["custom_episodes"], custom_episodes=[test_episode], \
                     n_targets=fmri[list(fmri.keys())[0]].shape[1])
            fmri_val_pred = model.predict(features_val[:,:]);
            cortex_target = pca.inverse_transform(fmri_val)
            cortex_predic = pca.inverse_transform(fmri_val_pred)
            with np.errstate(invalid='ignore'): # because of medial wall ==0
                bwscore= corr_score(cortex_target, cortex_predic)
            
            del cortex_target
            del cortex_predic
            encoding_accuracy, _ = compute_encoding_accuracy(fmri_val, fmri_val_pred)
            scores_bw_epi[test_episode] = bwscore
            scores_pc_epi[test_episode] = encoding_accuracy

        
        results = dict(scores_bw_mvset=scores_bw_mvset, scores_pc_mvset=scores_pc_mvset, \
            scores_bw_epi=scores_bw_epi, scores_pc_epi=scores_pc_epi);

        
        np.save(score_fn, results)
        print("saved:", score_fn)


def significance_testing(root_data_dir, feature_set, subject_nrs, enc_features, vstim="figures", args={}):
    print("\n\n" + "----" * 20)
    print("Significance testing of the average model")

    pca_path = "/scratch-scc/users/robert.scholz2/full_sample_pca91k.500comps.scaled.pkl"
    pca = pk.load(open(pca_path,'rb'))

    enc_features = {"visual": enc_features}

    for subject in subject_nrs:
        sub = f"sub-0{subject}"

        sign_fn = f"{root_data_dir}/ann_brain_data/scores/full_brain_models.{feature_set}.s1to5.{sub}.significance.avgmodel.npy"
        if os.path.exists(sign_fn):
            print(f"Skipping {sub} as exists already:", sign_fn)
            continue;

        fmri_fn = f"{root_data_dir}/ann_brain_data/mri/{sub}_allmovies_gcmpca.npy"
        fmri = np.load(fmri_fn, allow_pickle=True).item()
        fmri = {k.replace("_run-1", ""): v for k, v in fmri.items() if not k.endswith("_run-2")}
        # resulting keys: 'life05', 's01e01a', 's01e01b', 

        # **Load the saved average model**
        avg_model_fn = f"{root_data_dir}/ann_brain_data/models/full_brain_models.{feature_set}.s1to5.{sub}.avgmodel.pkl"
        if not os.path.exists(avg_model_fn):
            print(f"Skipping {sub} - Model not found: {avg_model_fn}")
            continue

        model = pk.load(open(avg_model_fn, "rb"))
        print(f"// {sub} - {feature_set}")

        ######## turn into parameter later
        #vstim = "figures" 
        n_perms = 5000 if not hasattr(args, "n_perms") else args.n_perms
        use_every_nth_tr= 10 if not hasattr(args, "use_every_nth_tr")  else args.use_every_nth_tr 
        rand_seed=189 if not hasattr(args, "rand_seed") else args.rand_seed

        features_val, fmri_val = align_features_and_fmri_samples(
                enc_features, fmri, 5, 5, HRF_DELAY_MIN_IN_TRs, STIM_WINDOW_IN_TRs,
                [vstim], n_targets=fmri[list(fmri.keys())[0]].shape[1])
        
        fmri_val_pred = model.predict(features_val[::use_every_nth_tr])
        cortex_target = pca.inverse_transform(fmri_val[::use_every_nth_tr])[:,:59412]
        cortex_predic = pca.inverse_transform(fmri_val_pred)[:,:59412]

        with np.errstate(invalid="ignore"):
            #bwscore = corr_score(cortex_target[::use_every_nth_tr], cortex_predic[::use_every_nth_tr])
            bwscore = fast_corr(cortex_target, cortex_predic)

        #cortex_predic=cortex_predic[::use_every_nth_tr]
        #cortex_target=cortex_target[::use_every_nth_tr]
        
        permuted_scores = []  
        np.random.seed(rand_seed)
        for i in tqdm(range(n_perms)):  
            idxs = compute_block_perm_indices(cortex_target.shape[0], block_size=10);  
            pred_perm = cortex_predic[idxs]  # (T, V)  
            perm_score = fast_corr(pred_perm, cortex_target)  # (V,)  
            permuted_scores.append(perm_score) 

        permuted_scores=np.array(permuted_scores)
        tvals, pvals = ttest_1samp(permuted_scores, popmean=bwscore[:59412], axis=0) # pvals.shape (59412,)
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_by')
        results = dict(permuted_scores=permuted_scores, pvals_corrected=pvals_corrected, pvals=pvals, bwscore=bwscore);
        
        np.save(sign_fn, results)
        print("saved:", sign_fn)



def regress_and_score_1000_parcel_control(root_data_dir, feature_set, subject_nrs, enc_features, all_training_keys):
    from brainannlib.algonauts_funcs import load_stimulus_features, load_fmri
    
    enc_features = {"visual": enc_features}

    for subject in subject_nrs:
        sub=f"sub-0{subject}"
        fmri = load_fmri(root_data_dir, subject)
        fmri = {k.replace("_run-1", ""): v for k, v in fmri.items() if not k.endswith("_run-2")}
        print(f"frmi for subj {sub} has {len(fmri.keys())} episodes");

        ############################
        ## Model Training 
        existing_training_keys=[epi for epi in all_training_keys if epi in fmri.keys()]
        if len(existing_training_keys) < len( all_training_keys):
            missing= [epi for epi in all_training_keys if not(epi in fmri.keys())]
            print("missing fmri-data for episodes:", missing)
        all_training_keys =np.array(existing_training_keys);
        
        # just use the first fold?
        kf = KFold(n_splits=5)
        train, test = list(kf.split(all_training_keys))[0];

        score_fn = f"{root_data_dir}/ann_brain_data/scores/full_brain_models.{feature_set}.s1to5.{sub}.scores.1000_parcel_validation.npy"
        if os.path.exists(score_fn):
            print("skipped. Already exists:", score_fn)
            continue;

        features_train, fmri_train = align_features_and_fmri_samples(
            enc_features, fmri, 5, 5, HRF_DELAY_MIN_IN_TRs, STIM_WINDOW_IN_TRs, ["custom_episodes"], 
            n_targets=fmri[list(fmri.keys())[0]].shape[1],
            custom_episodes=all_training_keys[train]
        )
        print(features_train.shape, fmri_train.shape, end=" ... ")
        model = train_sklearn_ridgecv(features_train, fmri_train)

        # pk.dump(model, open(model_fn, "wb"))
        # print(f"saved as :", model_fn)

        ############################
        ## Model scoring

        # model = pk.load(open(avg_model_fn, "rb"))
        print(f"// {sub} - {feature_set}")
        scores_bw_mvset, scores_pc_mvset = {}, {}

        scores_parcel_mvset={}
        # check if we do it for all, or just for figures
        for vstim in tqdm(all_movie_sets, total=len(all_movie_sets)):
            features_val, fmri_val = align_features_and_fmri_samples(
                enc_features, fmri, 5, 5, HRF_DELAY_MIN_IN_TRs, STIM_WINDOW_IN_TRs,
                [vstim], n_targets=fmri[list(fmri.keys())[0]].shape[1])
            fmri_val_pred = model.predict(features_val)

            with np.errstate(invalid="ignore"):
                parcel_score = corr_score(fmri_val, fmri_val_pred)

            scores_parcel_mvset[vstim] = parcel_score
            is_test = vstim not in train_movies
            print(f"{vstim:<15} {parcel_score.mean():.3f}  {'test' if is_test else 'train'}")
        
        # save scores ....
        results=dict(scores_parcel_mvset=scores_parcel_mvset)
        
        np.save(score_fn, results);
        print("Saved:", score_fn)
    return; 


if __name__ == "__main__":
    print(os.getpid())

    root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
    actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    featred_dir = f"{root_data_dir}/ann_brain_data/featred"


    parser = argparse.ArgumentParser(description="Process DNN activation features.")
    parser.add_argument("--model_name", type=str, default="SmolLM2-1.7B", help="Model name")
    parser.add_argument("--postfix", type=str, default="SmolLM2-ext", help="Feature set name")
    parser.add_argument("--subject_nrs", type=int, nargs="+", default=[1], help="Subject numbers")

    parser.add_argument("--feature_path", type=str, default=None, help="Use a sepcifc path to a cummulated stimulus file" +
                        " instead of finding it based on the postfix. The outputs will still be saved using the postfix.")

    parser.add_argument("--fit", action="store_true", help="Fit the models")
    parser.add_argument("--score", action="store_true", help="Score the estimated models")
    parser.add_argument("--fit_1000_parcel_control", action="store_true", help="Fit the 1000-parcel prediction model")
    parser.add_argument("--sign_testing", action="store_true", help="Do significance testing (based on every_nth_tr of figures movie)")
    
    parser.add_argument("--use_every_nth_tr", type=int, default=10, help="")
    parser.add_argument("--n_perms", type=int, default=5000, help="")
    parser.add_argument("--rand_seed", type=int, default=189, help="")

    parser.add_argument("--n_feat", type=int, default=2000, help="")


    args = parser.parse_args()

    print("\n\n" + "----" * 20)
    setup_environment()
    print("Arguments given:", args)

    model_name, postfix, subject_nrs = args.model_name, args.postfix, args.subject_nrs

    all_movie_sets = [
        "friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05",
        "friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life",
    ]
    train_movies = all_movie_sets[:5]
    print(train_movies)

    fn= f"{featred_dir}/actv-{model_name}.{postfix}.all_stimuli.s7ext.pca2000.npy"
    
    #############################################
    if args.feature_path:
        print("Load stimuli based on the given feature path:", args.feature_path)
        fn = args.feature_path
    #############################################

    all_features_dict = np.load(fn, allow_pickle=1).item();
    # keys: 'life05', 'friends_s01e01a', 'friends_s01e01b', 

    # **Load training data** (ensure these variables are properly initialized)
    all_features_dict = {k.split("_")[-1]:v for k,v in all_features_dict.items()}
    if not(args.n_feat is None):
        print("Only use (up to) the first", args.n_feat, "features.")
        all_features_dict = {k:v[:,:args.n_feat] for k,v in all_features_dict.items()}
    # resulting keys: 'life05', 's01e01a', 's01e01b', 
    prefixes = [mvset.split("-")[-1] for mvset in train_movies] # e.g. life05, s01, s05 ...
    all_training_keys = [k for k in all_features_dict.keys() if \
            any(k.startswith(pref) for pref in prefixes)] 
    all_training_keys=np.array(all_training_keys)
    print(len(all_training_keys), " of ", len(all_features_dict.keys()))#, all_training_keys)

    #
    # remove here missing episode for each subject (missing fmri)?
    # or inside estim, and score average?
    #

    feature_set=f"{model_name}.{postfix}"
    if args.fit:
        estimate_models(root_data_dir, feature_set, subject_nrs, all_features_dict, all_training_keys)
        generate_average_models(root_data_dir, feature_set, subject_nrs)
    
    if args.score:
        score_average_model(root_data_dir, feature_set, subject_nrs, all_features_dict, all_movie_sets, train_movies)

    if args.fit_1000_parcel_control:
        regress_and_score_1000_parcel_control(root_data_dir, feature_set, subject_nrs, all_features_dict, all_training_keys)


    if args.sign_testing:
        significance_testing(root_data_dir, feature_set, subject_nrs, all_features_dict, vstim="figures", args=args)
