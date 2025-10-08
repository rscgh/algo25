

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

#!hostname


root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
featred_dir = f"{root_data_dir}/ann_brain_data/featred"
model_dir =f"{root_data_dir}/ann_brain_data/models"
pred_dir = f"{root_data_dir}/ann_brain_data/preds"


def setup_environment():
    """Set up environment"""
    import socket
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())


def compile_features(args):

    enc_features={}
    stim_windows={}
    for model_def in args.embedding_comb.split(";"):

        # model_def e.g. "llama-3b:10030:3:1000"
        model_name, postfix, stim_window, n_feat = model_def.split(":")
        stim_window = int(stim_window); n_feat=int(n_feat);

        fn= os.path.join(featred_dir, f"actv-{model_name}.{postfix}.all_stimuli.s7ext.pca2000.npy");

        if not(os.path.exists(fn)):
            if model_name=="algo_slowr50":
                print("loading (hardcoded):")
                mod_features = load_stimulus_features(root_data_dir, ["visual"])['visual']
                mod_features = {k:v[:,:n_feat] for k,v in mod_features.items()}
                print(model_def, mod_features["s01e01a"].shape)
            else:
                print("Error: couldnt find", fn)
                print("So ignoring:", model_def)

        else:
            print("loading:", model_def)
            mod_features = np.load(fn, allow_pickle=True).item()
            mod_features = {k.split("_")[-1]: v for k,v in mod_features.items()}
            mod_features = {k:v[:,:n_feat] for k,v in mod_features.items()}
            print(model_def, mod_features["s01e01a"].shape)

        enc_features[model_def] = mod_features;
        stim_windows[model_def] = stim_window;

    return enc_features, stim_windows


def regress_algo25(enc_features, fmri, movies_train, subject, stimulus_window, args):
    # default params
    modality = "all";hrf_delay = 3; 
    excluded_samples_start, excluded_samples_end = 5, 5

    ### New ###################################################

    from brainannlib.ann_activations import load_ann_embeddings
    from brainannlib.regressions import delay_by_hrf, get_regression_pipeline
    from brainannlib.mri_data_loading import load_algonauts25_competition_training_mri_data
    #subject = 1 
    fmri = load_algonauts25_competition_training_mri_data(subject, root_data_dir=root_data_dir)
    brain_train =  np.concatenate([fmri[stim] for stim in train_stimuli], axis=0)[:-1]
    brain_test =  np.concatenate([fmri[stim] for stim in test_stimuli], axis=0)
    print("brain_data:", brain_train.shape, brain_test.shape)

    ann_embd_train = load_ann_embeddings(model_name, pretrained, dataset, train_stimuli, layers, v=0, mode=embd_mode)[layer_name]
    ann_embd_test = load_ann_embeddings(model_name, pretrained, dataset, test_stimuli, layers, v=0, mode=embd_mode)[layer_name]

    hrf_delays=[2,3,4]
    ann_embd_train = delay_by_hrf(ann_embd_train, delays=hrf_delays)
    ann_embd_test = delay_by_hrf(ann_embd_test, delays=hrf_delays)
    print("embeddings:", ann_embd_train.shape, ann_embd_test.shape)

    pipeline = get_regression_pipeline("RidgeCV") 
    _ = pipeline.fit(ann_embd_train, brain_train)

    #brain_pred = pipeline.predict(ann_embd_test)
    #encoding_accuracy, mean_encoding_accuracy = compute_encoding_accuracy(brain_test, brain_pred)

    ###########################################################
    """# old
    features_train, fmri_train = align_features_and_fmri_samples_v2(enc_features, fmri, excluded_samples_start, 
            excluded_samples_end, hrf_delay, stimulus_window, movies_train)
        
    feature_set=args.comb_name;
    model_fn = f"{model_dir}/full_brain_models.{feature_set}.{args.training_src}.sub-0{subject}.singlemodel.pkl"
    if os.path.exists(model_fn):
        print("loading existing model:", model_fn)
        model=pk.load(open(model_fn, "rb"));
        return model, features_train.shape;

    print("Training on:", features_train.shape, fmri_train.shape)
    model = train_sklearn_ridgecv(features_train[:, :], fmri_train);
    pk.dump(model, open(model_fn, "wb"))
    print("Saved model:", model_fn)
    return model, ann_embd_train.shape;

    """
    return pipeline[-1], ann_embd_train.shape;


def score_algo25(model, movies_val, fmri, enc_features, train_feat_shape, subject, stimulus_window, args):
    print("Validate on", movies_val)
    # default params
    modality = "all";hrf_delay = 3; 
    excluded_samples_start, excluded_samples_end = 5, 5
    scores=[]
    for mv in movies_val:
        features_val, fmri_val = align_features_and_fmri_samples_v2(enc_features, fmri, excluded_samples_start, 
                excluded_samples_end, hrf_delay, stimulus_window, [mv])
        fmri_val_pred = model.predict(features_val[:,:]);
        encoding_accuracy, mean_acc = compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality)
        scores.append([args.training_src, mv, train_feat_shape, mean_acc, encoding_accuracy])
        print("Scores:", scores[-1][:4])
    return scores;

def create_submission(enc_features, stimulus_window, args):
    print("Creating submission predictions", end="...")

    submission_predictions = {}
    s7feats= align_features_friends_s7_v2(root_data_dir, enc_features, stimulus_window, subjects = [1, 2, 3, 5])

    for s in [1, 2, 3, 5]:
        sub = f"sub-0{s}"
        print(s, end=" ")
        submission_predictions[sub] = {}
        feature_set=args.comb_name;
        model_fn = f"{model_dir}/full_brain_models.{feature_set}.{args.training_src}.sub-0{s}.singlemodel.pkl"
        if not os.path.exists(model_fn):
            print("Error: missing model for", sub, feature_set, args.training_src, model_fn)
        model=pk.load(open(model_fn, "rb"));

        for movie, mvfeat in s7feats[sub].items():
            # Predict fMRI responses for the aligned features of this episode, and
            fmri_pred = model.predict(mvfeat).astype(np.float32)
            submission_predictions[sub][movie] = fmri_pred
            if movie=="s07e03b": print(f"predictions for {sub} episode {movie}:", fmri_pred.shape)
    
    # Save the predicted fMRI dictionary as a .npy file
    output_file = f"{pred_dir}/model_comb_{args.comb_name}_trained_on_{args.training_src}-fmri_predictions_friends_s7.npy"
    np.save(output_file, submission_predictions)
    print(f"Formatted predictions saved to: {output_file}")

    # Zip the saved file for submission
    #zip_file = save_dir + f"fmri_predictions_friends_s7.{suffix}.zip"
    zip_file = f"{pred_dir}/model_comb_{args.comb_name}_trained_on_{args.training_src}-fmri_predictions_friends_s7.zip"
    import zipfile
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Submission file successfully zipped as: {zip_file}")
    os.remove(output_file);


def get_train_val_movies(args):

    #all_movie_sets = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", \
    #              "friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]
    movies_train=[]
    if args.training_src == "friends_s1-s5":
        movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05"]

    elif args.training_src == "all_minus_s06+fig":
        movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", \
                    "movie10-bourne", "movie10-wolf", "movie10-life"]

    elif args.training_src == "all":
        movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", \
                    "friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]

    movies_val = [mv for mv in ["friends-s06", "movie10-figures"] if not(mv in movies_train)]

    return movies_train, movies_val;
# windows: 1 3 4 5
# 500, 1000, 2000

if __name__ == "__main__":
    print(os.getpid())

    #root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
    #actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    #featred_dir = f"{root_data_dir}/ann_brain_data/featred"


    parser = argparse.ArgumentParser(description="Process DNN activation features.")
    parser.add_argument("--embedding_comb", type=str, default="Llama-3.1-8B:4L7T1000W:3:1000",
                         help="Model definitions separated by ';' in format 'name:postfix:stim_window:n_features'" \
                         "e.g. 'Llama-3.1-8B:4L7T1000W:3:1000'")
    parser.add_argument("--subject_nrs", type=int, nargs="+", default=[1], help="Subject numbers")
    parser.add_argument("--training_src", type=str, default="friends_s1-s5", help="....'all'")
    parser.add_argument("--comb_name", type=str, default="bestForSure", help="....'all'")
    parser.add_argument("--save", action="store_true", help="Save the estimated models")
    parser.add_argument("--prep_submission", action="store_true", help="Save the estimated models")

    args = parser.parse_args()

    print("\n\n" + "----" * 20)
    setup_environment()
    print("Arguments given:", args)

    embedding_comb=args.embedding_comb.split(";")
    enc_features, stim_window = compile_features(args)
    if len(enc_features.keys())<=0:
        raise FileExistsError("No encoding features could be loaded. Abort")

    print(enc_features.keys())
    print(stim_window)
    movies_train, movies_val = get_train_val_movies(args);

    for subject in args.subject_nrs:
        fmri = load_fmri(root_data_dir, subject) # subject = 1; 
        model, train_feat_shape = regress_algo25(enc_features, fmri, movies_train, subject, stim_window, args)
        scores = score_algo25(model, movies_val, fmri, enc_features, train_feat_shape, subject, stim_window, args)

        if len(embedding_comb)==1:
            # single model case
            single_embds_results_fn = "algo25_single_embds_results.csv"
            cols=["model", "postfix", "stim_window", "n_feat", "sub", "train", "test", "train_feat_shape", "mean_acc"] + list(np.arange(1000))
            df = pd.read_csv(single_embds_results_fn) if os.path.exists(single_embds_results_fn) else pd.DataFrame(columns=cols)
            print("Single embeddings df of len", len(df))
            
            id_cols = cols[:7]  # Columns to check as identifier
            nc = len(id_cols);

            model_def = args.embedding_comb.split(";")[0]
            model_name, postfix, stim_window, n_feat = model_def.split(":")
            
            for res in scores:
                train, test, train_feat_shape, mean_acc, encoding_accuracy = res
                new_row= model_def.split(":") + [subject] + [train, test, train_feat_shape, mean_acc] + list(encoding_accuracy);
                
                idx = df.loc[(df[id_cols] == new_row[:nc]).all(axis=1)].index
                print(idx, new_row[:len(new_row)-1000])

                if len(idx)>0: df.loc[idx] = new_row # replace existing row
                else: df.loc[len(df)] = new_row # add new row

            print("Saving to:", single_embds_results_fn)
            df.to_csv(single_embds_results_fn, index=False)


        else: # use a combination
            comb_embd_results_fn = "algo25_combined_embds_results.csv"

            cols=["model_combination", "sub", "train", "test", "train_feat_shape", "mean_acc"] + list(np.arange(1000))
            df = pd.read_csv(comb_embd_results_fn) if os.path.exists(comb_embd_results_fn) else pd.DataFrame(columns=cols)
            print("Combined embeddings df of len", len(df))
            
            id_cols = cols[:4]  # Columns to check as identifier
            nc = len(id_cols);
            
            for res in scores:
                train, test, train_feat_shape, mean_acc, encoding_accuracy = res
                new_row= [args.embedding_comb] + [subject] + [train, test, train_feat_shape, mean_acc] + list(encoding_accuracy);
                idx = df.loc[(df[id_cols] == new_row[:nc]).all(axis=1)].index
                
                print(idx, new_row[:len(new_row)-1000])
                if len(idx)>0: df.loc[idx] = new_row # replace existing row
                else: df.loc[len(df)] = new_row # add new row

            print("Saving to:", comb_embd_results_fn)
            df.to_csv(comb_embd_results_fn, index=False)
    
    if args.prep_submission:
        create_submission(enc_features, stim_window, args);
    