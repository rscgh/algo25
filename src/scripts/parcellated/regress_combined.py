######################################################################################
## This is the script that generates predictions based on mutlimodel features
## and makes them ready for submission on codabench
######################################################################################


import os, sys
import numpy as np
import pickle as pk
import zipfile

from brainannlib.algonauts_funcs import load_stimulus_features, load_fmri, align_features_and_fmri_samples
from brainannlib.algonauts_funcs import compute_encoding_accuracy
from brainannlib.algonauts_funcs import train_sklearn_ridgecv

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"] if "ALGONAUTS_ROOT_DIR" in os.environ.keys() else "/scratch-scc/users/robert.scholz2/cneuromod"
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"

all_movie_sets = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", \
                  "friends-s06", "movie10-bourne", "movie10-wolf", "movie10-figures", "movie10-life"]

# default params
subject = 1; modality = "all" 
excluded_samples_start, excluded_samples_end = 5, 5
hrf_delay, stimulus_window = 3, 5

# load the language features (tutorial had: 250)
fn= os.path.join(acc_dir, f"actv-SmolLM2-1.7B-algonauts_all_train-last5trs.pca2000.npy");
lang_features = np.load(fn, allow_pickle=True).item()
lang_features = {k.split("_")[-1]: v for k,v in lang_features.items()}
lang_features = {k:v[:,:500] for k,v in lang_features.items()}
print("language:", lang_features["s01e01a"].shape)

# load the audio features (tutorial had: 20)
fn= os.path.join(acc_dir, f"actv-whisper-small-algonauts_all_train-2sCNKperTR-last4token.pca2000.npy");
audio_features = np.load(fn, allow_pickle=True).item()
audio_features = {k.split("_")[-1]: v for k,v in audio_features.items()}
audio_features = {k:v[:,:500] for k,v in audio_features.items()}
print("audio:", audio_features["s01e01a"].shape)

# load visual features (tutorial had: 250)
#fn= os.path.join(acc_dir, f"actv-clipvit-base-patch32-algonauts_all_train.cls-sc.npy");
#features_visual= np.load(fn, allow_pickle=1).item()["visual"]
features_visual = load_stimulus_features(root_data_dir, ["visual"])['visual']
print("visual:", features_visual["s01e01a"].shape)

# put them all in a single dict to do algonaut-tutorial-style regression 
# using their functions
enc_features ={"language":lang_features,"audio":audio_features, "visual":features_visual}


# use all seasons/full-movies as training data
movies_train = all_movie_sets; 
models={}

# Fitting the model for each of the subjects 
for s in [1, 2, 3, 5]:
      sub = f"sub-0{s}"
      print(f"### {sub} ###")

      fn = acc_dir+f"/model_comb_smolwhispr50_{sub}_trained_on_all.pkl"
      #if os.path.exists(fn): continue; # skip if already exisits
      
      # loading subject specific MRI data
      fmri = load_fmri(root_data_dir, s)
      # align features and mri-data
      features_train, fmri_train = align_features_and_fmri_samples(enc_features, fmri, excluded_samples_start, 
            excluded_samples_end, hrf_delay, stimulus_window, movies_train)

      print("Training on:", features_train.shape, fmri_train.shape)
      model = train_sklearn_ridgecv(features_train[:, :], fmri_train);

      # saving the regression-model as pkl, so it can be readily loaded later on
      pk.dump(model, open(fn,"wb"))
      print("Saved:", fn)
      models[sub] = model;
      
      # for each of the seasons/full-movies, check how well out model is doing
      for vstim in all_movie_sets:
            features_val, fmri_val = align_features_and_fmri_samples(enc_features, fmri, excluded_samples_start, 
                  excluded_samples_end, hrf_delay, stimulus_window, [vstim])

            fmri_val_pred = model.predict(features_val[:,:]);
            encoding_accuracy, mean_encoding_accuracy = compute_encoding_accuracy(fmri_val, fmri_val_pred, s, modality)
            
            # test-set (i.e. non trained on)-seasons/movies are marked in bold (if any)
            if not(vstim in movies_train): print("\033[1m", end="")
            print(vstim, (15-len(vstim))*" ", mean_encoding_accuracy, "  ", "train" if vstim in movies_train else "test\033[0m")
      


#fn= os.path.join(acc_dir, f"models_comb_smolwhispr50_trained_on_all.npy");
#np.save(fn, models)

# align_features_friends_s7 is just a more compact function for preparation of the test data features
from brainannlib.algonauts_funcs import align_features_friends_s7
print("Preprocessing features ...")
s7feats= align_features_friends_s7(root_data_dir, enc_features)
print(len(s7feats.keys()), s7feats.keys())
print(s7feats["sub-01"]['s07e01a'].shape)

# generate the predictions for friends season7
submission_predictions = {}
for s in [1, 2, 3, 5]:
    sub = f"sub-0{s}"
    submission_predictions[sub] = {}

    for movie, mvfeat in s7feats[sub].items():
        # Predict fMRI responses for the aligned features of this episode, and
        fmri_pred = models[sub].predict(mvfeat).astype(np.float32)
        submission_predictions[sub][movie] = fmri_pred
        if movie=="s07e03b": print(f"predictions for {sub} episode {movie}:", fmri_pred.shape)


# Save the predicted fMRI dictionary as a .npy file
output_file = acc_dir + "/model_comb_smolwhispr50_trained_on_all-fmri_predictions_friends_s7.npy"
np.save(output_file, submission_predictions)
print(f"Formatted predictions saved to: {output_file}")

# Zip the saved file for submission
#zip_file = save_dir + f"fmri_predictions_friends_s7.{suffix}.zip"
zip_file = acc_dir + "/model_comb_smolwhispr50_trained_on_all-fmri_predictions_friends_s7.zip"
with zipfile.ZipFile(zip_file, 'w') as zipf:
    zipf.write(output_file, os.path.basename(output_file))
print(f"Submission file successfully zipped as: {zip_file}")
os.remove(output_file)