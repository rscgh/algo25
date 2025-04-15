
######################################################################################
## This class contains the relevant functions from the Algonauts2025 tutorial 
## located at: https://colab.research.google.com/drive/1fop0zvaLBLBagvJRC-HDqGDSgQElNWZB
## along with slightly modified/adapted functions
######################################################################################

import h5py, os
import numpy as np
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
root_data_dir = "/scratch-scc/users/robert.scholz2/cneuromod/"
from tqdm.auto import tqdm



def load_fmri(root_data_dir, subject, v=False, average_repeat_runs=True):
    """
    Load the fMRI responses for the selected subject.
    returns dictionary containing the  fMRI responses with stimulus/movie names as key.
    """
    fmri = {}
    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors','fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    keys = fmri_friends.keys();
    if v: print("fmri_friends:", len(keys), list(keys)[:5] )

    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors','fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    keys = fmri_movie10.keys();
    if v: print("fmri_movie10:", len(keys), list(keys)[:5] )
    
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10

    if average_repeat_runs:

        # Average the fMRI responses across the two repeats for 'figures'
        keys_all = fmri.keys()
        figures_splits = 12
        for s in range(figures_splits):
            movie = 'figures' + format(s+1, '02')
            keys_movie = [rep for rep in keys_all if movie in rep]
            fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]
        # Average the fMRI responses across the two repeats for 'life'
        keys_all = fmri.keys()
        life_splits = 5
        for s in range(life_splits):
            movie = 'life' + format(s+1, '02')
            keys_movie = [rep for rep in keys_all if movie in rep]
            fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]

    ### Output ###
    keys = fmri.keys();
    if v: print("fmri:", len(keys), list(keys)[:5] )
    return fmri



def load_stimulus_features(root_data_dir, modality):
    features = {}
    ### Load the visual features ###
    if modality == 'visual' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', 'visual', 'features_train.npy')
        features['visual'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the audio features ###
    if modality == 'audio' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', 'audio', 'features_train.npy')
        features['audio'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Load the language features ###
    if modality == 'language' or modality == 'all':
        stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', 'language', 'features_train.npy')
        features['language'] = np.load(stimuli_dir, allow_pickle=True).item()

    ### Output ###
    return features


def load_stimulus_features_friends_s7(root_data_dir):
    features_friends_s7 = {}

    ### Load the visual features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'visual', 'features_test.npy')
    features_friends_s7['visual'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the audio features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'audio', 'features_test.npy')
    features_friends_s7['audio'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the language features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'language', 'features_test.npy')
    features_friends_s7['language'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Output ###
    return features_friends_s7






'''
def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies, v=False):
    """
    Align the stimulus feature with the fMRI response samples for the selected
    movies, later used to train and validate the encoding models.
    """

    ### Empty data variables ###
    aligned_features = []
    aligned_fmri = np.empty((0,1000), dtype=np.float32)
    isfirst = True;
    
    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == 'friends':
            id = movie[8:]
        elif movie[:7] == 'movie10':
            id = movie[8:]

        #print(movie, id)
        movie_splits = [key for key in fmri if id in key[:len(id)]]
        if v==1: print(movie, id, movie_splits)
        #movie_splits = [key for key in fmri if id in key]

        if v==2: print(f"#movie {movie}, id {id}, n_movie_splits: {len(movie_splits)} e.g. {movie_splits}")
        
        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split_x = fmri[split]
            # Exclude the first and last fMRI samples
            fmri_split = fmri_split_x[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            if isfirst and v==2: 
                print("\nsplit", split, len(fmri_split), fmri_split_x.shape, fmri_split.shape, aligned_fmri.shape)
                #isfirst=False;

            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):
                # Empty variable containing the stimulus features of all
                # modalities for each fMRI sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay \
                                - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > (len(features[mod][split])):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                        if isfirst and (s in [0, 1, 2, 10, 30]) and v==2:
                            print(mod, f"s={s}, idx_start={idx_start}, idx_end={idx_end}, idx_max={len(features[mod][split])}, f={f.shape}, {f.flatten().shape}")

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        idx = s + excluded_samples_start - hrf_delay
    	                
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1,:]
                        else:
                            f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())

                        if isfirst and (s in [0, 1, 2, 10, 30]) and v==2:
                            print(mod, f"s={s}, idx={idx}, idx_max={len(features[mod][split])}, f={f.shape}, {f.flatten().shape}")

                 ### Append the stimulus features of all modalities for this sample ###
                if isfirst and (s in [0, 1, 2, 10, 30]) and v==2:
                    print("f_all", f_all.shape)
                aligned_features.append(f_all)
        
            isfirst=False;

    ### Convert the aligned features to a numpy array ###
    aligned_features = np.asarray(aligned_features, dtype=np.float32)

    ### Output ###
    return aligned_features, aligned_fmri
''';

def align_features_and_fmri_samples_friends_s7(features_friends_s7,
    root_data_dir):

    n_samples = {}
    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}
    hrf_delay = 3
    stimulus_window = 5

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
            'fmri', f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            #print(epi, samples)
            features_epi = []

            ### Loop over fMRI samples ###
            for s in range(samples):
                # Empty variable containing the stimulus features of all
                # modalities for each sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features_friends_s7.keys():
                    if mod == 'visual' or mod == 'audio':

                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window

                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = idx_end - stimulus_window
                        f = features_friends_s7[mod][epi][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    elif mod == 'language':
                        idx = s - hrf_delay
                        if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                            f = features_friends_s7[mod][epi][-1,:]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        f_all = np.append(f_all, f.flatten())

                ### Append the stimulus features of all modalities for this sample ###
                features_epi.append(f_all)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = np.asarray(
                features_epi, dtype=np.float32)

            if not epi in n_samples.keys():
                n_samples[epi]=[]
            n_samples[epi].append(samples);
            
    for epi in n_samples.keys():
        if not np.all(n_samples[epi][0]==np.array(n_samples[epi])):
            print("Samples are off across subjects:", epi,  n_samples[epi])

    return aligned_features_friends_s7



def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject=None, modality=None):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p], fmri_val_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    return encoding_accuracy, mean_encoding_accuracy

def plot_accuracy_on_brain(encoding_accuracy, mean_encoding_accuracy, subject, title, \
                           root_data_dir=root_data_dir, vmin=0,vmax=0.7):
    ### Map the prediction accuracy onto a 3D brain atlas for plotting ###
    atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join(root_data_dir, 'algonauts_2025.competitors', 'fmri', f'sub-0{subject}', 'atlas', atlas_file)
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, sub-0{subject}, {title}, mean accuracy: " + str(mean_encoding_accuracy)
    display = plotting.plot_glass_brain(
        encoding_accuracy_nii, display_mode="lyrz", cmap='hot_r', colorbar=True,
        plot_abs=False, symmetric_cbar=False, title=title, vmin=vmin, vmax=vmax )
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)
    plotting.show()


def train_sklearn_ridgecv(features_train, fmri_train, alpha_coef=1, **kwargs):
    alphas = np.asarray((1000000, 100000, 10000, 1000, 100, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001))*alpha_coef
    defkwargs = dict(alphas=alphas, cv=None, alpha_per_target=True)
    defkwargs.update(kwargs)
    model = RidgeCV(**defkwargs)
    model.fit(features_train, fmri_train)    
    return model


movies_train = ["friends-s02", "friends-s03", "friends-s04", "friends-s06"]

def train_linear_encoding(features_train, fmri_train):
    model = LinearRegression().fit(features_train, fmri_train)
    return model

root_data_dir = "/scratch-scc/users/robert.scholz2/cneuromod/"

def run_encoding(features,fmri, modality, subject, movies_train, movies_val,
                 train_encoding=train_linear_encoding, rdir=  root_data_dir,
                 excluded_samples_start = 5, excluded_samples_end=5, hrf_delay=3, 
                 stimulus_window = 5):
    if fmri is None: fmri= load_fmri(rdir, subject)
    enc_features = features if modality=="all" else {modality:features[modality]};
    features_train, fmri_train = align_features_and_fmri_samples(enc_features, fmri, excluded_samples_start, 
        excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    features_val, fmri_val = align_features_and_fmri_samples(enc_features, fmri, excluded_samples_start, 
        excluded_samples_end, hrf_delay, stimulus_window, movies_val)
    
    model = train_encoding(features_train, fmri_train);
    fmri_val_pred = model.predict(features_val);
    
    encoding_accuracy, mean_encoding_accuracy = compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality)
    return model, fmri_val_pred, fmri_val, encoding_accuracy, mean_encoding_accuracy


######################################################################################
## New and simplified functions:

def load_full_fmri(root_data_dir, subjects=[1, 2, 3, 5], v=0, average_repeats=True):
    """
    Load fMRI responses for both train and test data across all subjects.
    """
    fmri_data = {f'sub-0{s}':{} for s in subjects}
    for subject in subjects:
        fmrid={}
        fmri_path = os.path.join(root_data_dir, 'algonauts_2025.competitors', 'fmri', f'sub-0{subject}')
        
        end=f'func/sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        file = os.path.join(fmri_path, end)
        fmri_dataset = h5py.File(file, 'r')
        # n=292 e.g. ['ses-001_task-s01e02a', 'ses-001_task-s01e02b', 'ses-001_task-s01e03a', 'ses-001_task-s01e03b'] ['ses-065_task-s06e24a', 'ses-066_task-s06e24b', 'ses-066_task-s06e24c', 'ses-066_task-s06e24d']
        keys=list(fmri_dataset.keys())
        if v==1: print(len(keys), [k[13:] for k in keys])
        for k in keys: fmrid[k[13:]] = fmri_dataset[k][:].astype(np.float32)
        fmri_dataset.close()

        end=f'func/sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
        file = os.path.join(fmri_path, end)
        fmri_dataset = h5py.File(file, 'r')
        # 61 ['ses-001_task-bourne01', 'ses-001_task-bourne02', 'ses-001_task-bourne03', 'ses-001_task-bourne04', 'ses-001_task-bourne05', 'ses-002_task-bourne06', 'ses-002_task-bourne07', 'ses-002_task-bourne08', 'ses-002_task-bourne09', 'ses-002_task-bourne10', 'ses-003_task-wolf01', 'ses-003_task-wolf02', 'ses-003_task-wolf03', 'ses-003_task-wolf04', 'ses-003_task-wolf05', 'ses-003_task-wolf06', 'ses-004_task-wolf07', 'ses-004_task-wolf08', 'ses-004_task-wolf09', 'ses-004_task-wolf10', 'ses-004_task-wolf11', 'ses-004_task-wolf12', 'ses-005_task-wolf13', 'ses-005_task-wolf14', 'ses-005_task-wolf15', 'ses-006_task-life01_run-1', 'ses-006_task-life02_run-1', 'ses-006_task-wolf16', 'ses-006_task-wolf17', 'ses-007_task-figures01_run-1', 'ses-007_task-figures02_run-1', 'ses-007_task-figures03_run-1', 'ses-007_task-life03_run-1', 'ses-007_task-life04_run-1', 'ses-007_task-life05_run-1', 'ses-008_task-figures05_run-1', 'ses-008_task-figures06_run-1', 'ses-008_task-figures07_run-1', 'ses-008_task-figures08_run-1', 'ses-008_task-figures09_run-1', 'ses-009_task-figures04_run-1', 'ses-009_task-figures10_run-1', 'ses-009_task-figures11_run-1', 'ses-009_task-figures12_run-1', 'ses-009_task-life01_run-2', 'ses-009_task-life02_run-2', 'ses-010_task-life03_run-2', 'ses-010_task-life04_run-2', 'ses-010_task-life05_run-2', 'ses-011_task-figures01_run-2', 'ses-011_task-figures02_run-2', 'ses-011_task-figures03_run-2', 'ses-011_task-figures04_run-2', 'ses-012_task-figures05_run-2', 'ses-012_task-figures06_run-2', 'ses-012_task-figures07_run-2', 'ses-012_task-figures08_run-2', 'ses-012_task-figures09_run-2', 'ses-012_task-figures10_run-2', 'ses-012_task-figures11_run-2', 'ses-012_task-figures12_run-2']
        keys=list(fmri_dataset.keys())
        if v==1: print(len(keys), [k for k in keys])
        for k in keys: fmrid[k[13:]] = fmri_dataset[k][:].astype(np.float32)
        fmri_dataset.close()

        if average_repeats:
            # average across repeats if existent (that is the case for movie10 - figures and life):
            run_pattern = re.compile(r"(.+)_run-\d+")
            movie_repeats = {}
            for key in fmrid.keys():
                match = run_pattern.match(key)
                if match: movie_repeats.setdefault(match.group(1), []).append(key)

            for movie, runs in movie_repeats.items():
                if v>1: print(movie, len(runs), runs)
                fmrid[movie] = np.mean([fmrid[run] for run in runs], axis=0).astype(np.float32)
                for run in runs: del fmrid[run]
                                
        fmri_data[f'sub-0{subject}'].update(fmrid)

    return fmri_data


def load_stimulus_features(root_data_dir, modalities=['visual', 'audio', 'language']):
    """
    Load stimulus features for all modalities (visual, audio, language) from both train and test files.
    """
    features = {m:{} for m in modalities}
    for dataset in ['train', 'test']:
        for modality in modalities:
            path = os.path.join(root_data_dir, 'stimulus_features', 'pca',
                'friends_movie10', modality, f'features_{dataset}.npy')
            features[modality].update(np.load(path, allow_pickle=True).item())
    return features



def align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end,\
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
                    if mod in ['visual', 'audio']:
                        # the feature samples need all be of the same length (=stimulus_window) and
                        # are bound by [excluded_samples_start, episode_feat_len]
                        # are defined by reference to the current TR sample s 
                        # and as s is based on the truncted mri-signal
                        #    we have to add excluded_samples_start to get the matching stimulus TR
                        # lastly, the hrf-delay is factored in
                        # approx: [start]..stim_window..[end]..hrf-delay..[s]
                        idx_start = max(excluded_samples_start, s + excluded_samples_start - hrf_delay - stimulus_window + 1)
                        idx_end = idx_start + stimulus_window
                        if idx_end > episode_feat_len:
                            idx_end = episode_feat_len
                            idx_start = idx_end - stimulus_window

                        # we get the feature vectors for multiple samples (N=stimulus_window), and flatten them
                        f = mod_features[episode][idx_start:idx_end].flatten()
                        
                        if isfirst and (s in [0, 1, 2, 10, 30, n_trs-2, n_trs-1]) and v>=2:
                            print(mod, f"s={s}, idx_start={idx_start}, idx_end={idx_end}, idx_max={episode_feat_len}, f={f.shape}, {f.flatten().shape}")

                    else:  # 'language'
                        # similiar procedure as above with slightly different bound [0, episode_feat_len-1]
                        # and only get a single embedding
                        # approx: [idx]..hrf_delay..[s]
                        idx = max(0, s + excluded_samples_start - hrf_delay)
                        if idx >= episode_feat_len - hrf_delay: idx = -1
                        f = mod_features[episode][idx].flatten()
                    
                        if isfirst and (s in [0, 1, 2, 10, 30, n_trs-2, n_trs-1]) and v>=2:
                            print(mod, f"s={s}, idx={idx}, idx_max={episode_feat_len}, f={f.shape}, {f.flatten().shape}")

                    f_all = np.append(f_all, f)

                if isfirst and (s in [0, 1, 2, 10, 30, n_trs-2, n_trs-1]) and v>=2:
                    print(f"s={s}, f_all_shape={f_all.shape}")
                        
                aligned_features.append(f_all)
            
            isfirst=False;

    return np.array(aligned_features, dtype=np.float32), aligned_fmri


def align_features_friends_s7(root_data_dir, features, subjects = [1, 2, 3, 5]):
    hrf_delay, stimulus_window = 3, 5
    aligned_feautures, n_samples = {}, {}

    for sub in subjects:
        sub_key = f'sub-0{sub}'
        aligned_feautures[sub_key] = {}
        fmri_samples = np.load(
            os.path.join(
                root_data_dir, 'algonauts_2025.competitors', 'fmri', sub_key,
                'target_sample_number', f'{sub_key}_friends-s7_fmri_samples.npy'
            ), allow_pickle=True).item()

        for episode, n_trs in fmri_samples.items():
            features_epi = []
            for s in range(n_trs):
                f_all = []
                for mod, mod_features in features.items():
                    episode_feat_len= len(mod_features[episode])
                    if mod in ['visual', 'audio']:
                        idx_start = max(0, s - hrf_delay - stimulus_window + 1)
                        idx_end = idx_start + stimulus_window
                        if idx_end > episode_feat_len:
                            idx_end = episode_feat_len;
                            idx_start = idx_end - stimulus_window
                        f = mod_features[episode][idx_start:idx_end].flatten()
                    
                    else: # language
                        idx =s - hrf_delay #max(0, s - hrf_delay)
                        if idx >= (episode_feat_len - hrf_delay): idx = -1
                        f = mod_features[episode][idx].flatten()
                    
                    f_all.append(f)
                
                features_epi.append(np.concatenate(f_all))
            
            aligned_feautures[sub_key][episode] = np.array(features_epi, dtype=np.float32)
            n_samples.setdefault(episode, []).append(n_trs)

    for episode, n_trs in n_samples.items():
        if not np.all(n_trs[0] == np.array(n_trs)):
            print("Samples are off across subjects:", episode, n_trs)

    return aligned_feautures