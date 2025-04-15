##############################################################################
# Brain Data
##############################################################################

def get_subj_brain_data(subj, dataset, stim_names=None, path = None, zscore=True, hemi="left"):
    if stim_names is None:
        from lib.config import dataset_stimuli
        stim_names = list(dataset_stimuli[dataset])
    if path is None:
        from lib.config import datasets_paths
        path = datasets_paths[dataset]
    
    if dataset.startswith("NSD"):
        subj_data = load_nsd_brain_data_v2(subj, path)[:, :163842]
    elif dataset=="hcpmovies":
        subj_data = np.concatenate([get_hcp_movie_data(subj, mv, datasets_paths[dataset], zscore=zscore) for mv in stim_names], axis=0)
    elif dataset.startswith("narratives"):
        subj_data = np.concatenate([get_narrative_data(subj, stim_name, datasets_paths[dataset], zscore=zscore) for stim_name in stim_names], axis=0)
    elif dataset=="CNMFriendsS02e1to5":
        subj_data = np.concatenate([get_cneuro_movie_data(subj, stim_name, datasets_paths[dataset], zscore=zscore) for stim_name in stim_names], axis=0)

    #elif dataset.startswith("CNMAlgonauts"):
    #    subj_data = np.concatenate([get_cneuro_movie_data(subj, stim_name, datasets_paths[dataset], zscore=zscore) for stim_name in stim_names], axis=0)
    
    else: 
        raise Exception("Couldn't figure out what brain data to laod")

    subj_data = np.nan_to_num(subj_data)
    return subj_data;




import nibabel as nib
from scipy import stats
nib.imageglobals.logger.setLevel(40)

def get_hcp_movie_data(subj, movie_name, local_hcp_dir = "/scratch/users/robert.scholz2", \
                       bma_slice=slice(0, 29696, None), zscore=True, v=False, axis=0):
  name_to_id = {'7T_MOVIE1_CC1_v2': 'MOVIE1_7T_AP', '7T_MOVIE2_HO1_v2': 'MOVIE2_7T_PA', '7T_MOVIE3_CC2_v2': 'MOVIE3_7T_PA', '7T_MOVIE4_HO2_v2': 'MOVIE4_7T_AP'};
  movie_id = name_to_id[movie_name]
  file_path = local_hcp_dir + f"/HCP_1200/{subj}/MNINonLinear/Results/tfMRI_{movie_id}/tfMRI_{movie_id}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
  if v: print(file_path)
  nimg = nib.load(file_path)
  fsdata = nimg.get_fdata()[:, bma_slice]
  return stats.zscore(fsdata, axis=axis) if zscore else fsdata;



import numpy as np;
import nibabel as nib;
from scipy.stats import zscore;
import os;

def load_nsd_brain_data(subj , nsd_dir, n_sessions=20, behaviour_df=None, v=False):
    
  data_folder = f"{nsd_dir}/{subj}"
  
  fn= f"{nsd_dir}/{subj}/{subj}_braindata_nsd2536rep3.fsav.LR.npy" # about 3gb
  if os.path.exists(fn):
     return np.load(fn);

  # otherwise collect all the data (takes time)
    
  if v: print("Sess\t#Condi\t\tL+R ts_shape")
  betas = []
  for ses in range(n_sessions): #40
    ses_i = ses+1
    si_str = str(ses_i).zfill(2)
    if v: print(si_str, end="\t")
    # this_ses = nsda.read_behavior(subject=sub, session_index=ses_i)
    #  ses_conditions = np.asarray(this_ses['73KID'])
    if not(behaviour_df is None):
      session_behavior = behaviour_df[behaviour_df['SESSION'] == ses_i]
      ses_conditions = np.asarray(session_behavior['73KID'])
      if v: print(ses_conditions.shape, end="\t\t")
      if len(ses_conditions)==0: continue;

      
    fn = f"lh.fsav_fhrfGLMdenoise_betas_session{si_str}.mgh"
    img_lh = nib.load( os.path.join(data_folder,fn)).get_fdata().squeeze()
    fn = f"rh.fsav_fhrfGLMdenoise_betas_session{si_str}.mgh"
    img_rh = nib.load( os.path.join(data_folder,fn)).get_fdata().squeeze()
    if v: print(img_lh.shape, img_rh.shape)
    #img_rh = np.zeros_like(img_lh)
    
    #all_verts = np.vstack((img_lh, img_rh))
    all_verts = np.vstack((zscore(img_lh, axis=1), zscore(img_rh, axis=1)))
    #betas.append((zscore(all_verts, axis=1)).astype(np.float32))
    betas.append(all_verts.astype(np.float32))
    #betas.append((zscore(all_verts, axis=0)).astype(np.float32)) << doesnt change anything
    
  betas = np.array(betas)
  if v: print(betas.shape)
    
  print("betas.shape:", betas.shape)
  ts = np.nan_to_num(np.concatenate(np.array(betas), axis=1))
  del betas
  n_vox, n_stim = ts.shape
  if v: print("timeseries shape:", n_vox, n_stim) # 327684 15000

  # Load the image ids corresponding to the timeseries
  s1r = pd.read_csv(f"misc/nsd_{subj}resp.tsv",   delimiter="\t")
  beh=s1r['73KID'][:n_stim]
  v,c = np.unique(beh, return_counts=1)
  # get the ids of all images images were presented 3 times   (~max number of presentations)
  good_imgs = v[c==3]
  if v: print(len(good_imgs), f"images with 3 presentations of {len(v)} total unique images (in {n_stim} individual presentations).")
    
  from tqdm.auto import tqdm
  # find the indices for each of the three represtions of all images that were presented 3 times
  good_imge_rep_ids = np.zeros((len(good_imgs), 3))
  for i, img_id in tqdm(enumerate(good_imgs),   total=len(good_imgs)):
    good_imge_rep_ids[i,:] = np.where(beh == img_id)[0]

  # use reliable vertices in the brain and first 2000 images
  brain_data_target = ts[:,good_imge_rep_ids[:,0].astype(int)].T;
    
  np.save(fn, brain_data_target)

  return brain_data_target;


def load_nsd_brain_data_v2(subj , nsd_dir):
      
  single_file = f"{nsd_dir}/{subj}/{subj}_braindata_nsd907commonIMGs.fsav.LR.npy"
  return np.load(single_file);



from sklearn.preprocessing import StandardScaler
zscaler = StandardScaler(with_mean=True, with_std=True)

def get_narrative_data(subj, task , brain_part = "fsav6_left", zscore=True, ds_path="/scratch/users/robert.scholz2/narratives"):
  #if brain_part!="fsav6_left": 
  # this is to make sure the first scan was aquired with story start, so both line up
  # mri scan can be longer, so sometimes i choose to leave additional TRs at the end, eg. 8TS, due to BOLD response delay
  #global story_slices;
  #slc = story_slices[task] if task in story_slices else slice(None)
  slc = slice(None)
  simg = nib.load(f"{ds_path}/derivatives/afni-nosmooth/{subj}/func/{subj}_task-{task}_space-fsaverage6_hemi-L_desc-clean.func.gii")
  fsdata = np.array([simg.darrays[i].data for i in range(len(simg.darrays))])
  return zscaler.fit_transform(fsdata[slc]) if zscore else fsdata[slc];


def get_cneuro_movie_data(subj, episode_id, dataset_dir ="/scratch/users/robert.scholz2/cneuroprep", bma_slice=slice(0, 29696, None), zscore=True, v=False):
  from glob import glob
  datadir = dataset_dir+"/fmriprep/friends/"
  # e.g. sub-01/ses-001/func/sub-01_ses-001_task-s01e06a_space-fsLR_den-91k_bold.dtseries.nii
  filename=f"{subj}/ses-001/func/{subj}_ses-001_task-{episode_id}_space-fsLR_den-91k_bold.dtseries.nii"
  searchstring=dataset_dir+"/fmriprep/friends/"+f"{subj}/ses-*/func/*{episode_id}*space-fsLR_den-91k_bold.dtseries.nii";
  res = glob(searchstring)
  assert len(res)>0, f"Couldnt find the file for subject {subj} episode {episode_id} with glob pattern: {searchstring}"
  if v and (len(res)>=2): 
      print(f"Info: glob retruned >1 file for subject {subj} episode {episode_id} with glob pattern: {searchstring}");
  file_path = res[0]
  assert os.path.exists(file_path), f"Datalad file still needs to be downloaded: {file_path}"
    
  if v: print(file_path)
  if v: print(os.path.exists(file_path))
  if v: print(os.stat(file_path).st_size)
  nimg = nib.load(file_path)
  fsdata = nimg.get_fdata()[:, bma_slice]
  return stats.zscore(fsdata, axis = 0) if zscore else fsdata;













def load_algonauts25_competition_training_mri_data(subject, root_data_dir = '/scratch/users/robert.scholz2/cneuromod'
):
    """
    Load the fMRI responses for the selected subject.
    returns dictionary containing the  fMRI responses with stimulus/movie names as key.
    """
    import h5py
    fmri = {}
    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
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
    return fmri