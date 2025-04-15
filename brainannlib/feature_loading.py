

import numpy as np
"""
def load_and_concat_activations(file, layers=None, swapaxes=True):
    x=np.load(file, allow_pickle=1)
    if x.shape==(): 
        # if activations are extracted as output_hidden_states, 
        # we need to saw layer with timepoint axis before flatten & concat
        x=x.item(); 
        #swapaxes=True; 
    data = x;

    if isinstance(x, dict):
        if layers is None: 
            layers = list(x.keys())
         
            #if len(layers) >4: 
            #    indices = np.linspace(0, len(layers) - 1, 4, dtype=int)
            #   print(layers)
            #    layers = np.array(layers)[indices].tolist()
            #    print(indices, layers)

        for l in layers: 
            if not(l in x.keys()): print(f"Warning: missing layer {l} in file {file}")
        data = np.concatenate([v for k,v in x.items() if k in layers], axis=-1).astype(np.float32)
   
     # else subselect layers from an array?
    shp=data.shape;
    if swapaxes: 
        #data = np.swapaxes(x, 0,1)[:,:,-1,:]
        data = np.swapaxes(x, 0,1)[:,:,:,:] 
        # out: time, layer, token, hidden
        
    #data = data.reshape(data.shape[0], -1)
    data = data.astype(np.float32);
    return data, shp, layers; 
"""

def load_and_concat_activations(file, layers=None, swapaxes=True):
    x=np.load(file, allow_pickle=1)
    if x.shape==(): x=x.item(); 
    data = x;

    if isinstance(x, dict):
        if layers is None: 
            layers = list(x.keys())

        for l in layers: 
            if not(l in x.keys()): print(f"Warning: missing layer {l} in file {file}")
        data = np.stack([v for k,v in x.items() if k in layers]).astype(np.float32)
   
     # else subselect layers from an array?
    shp=data.shape;
    if swapaxes: 
        data = np.swapaxes(data, 0,1)
        # out: time, layer, token, hidden; or time, layer, ...
        
    data = data.astype(np.float32);
    print(data.shape)
    return data, shp, layers;



def quickalign(features, fmri, stim_names, type="stimulus-set", ret="dict", \
               ess=5, ese=5, hrf_delay=5, stim_wind=5, v=False):
    
    aligned_fmri = {}
    aligned_feat = {}

    if isinstance(stim_names, str): stim_names=[stim_names];
    for stimulus in stim_names:
        stim_fmri=[]
        stim_feat=[]

        if type == "episodes": episodes = [stimulus];
        else:
            # stimset e.g. "s01", "s02", ... "bourne", "life", ..."friends-s01", "movie10-bourne" ...
            stimulus = stimulus.split("-")[-1]
            episodes = [key for key in fmri if key.startswith(stimulus)]
        
        for ne, epi in enumerate(episodes):
            fmri_run = fmri[epi][ess:-ese]
            n_trs = len(fmri_run)
            stim_fmri.append(fmri_run)

            # for each tr in the fmri-data
            for s in range(n_trs):
                tr_feat = []
                # dont discriminate by modality
                for raw_feats in features:
                    episode_feat_len= len(raw_feats[epi])

                    """
                    idx_start = max(0, s + ess - hrf_delay - stim_wind)
                    #idx_start = max(0, s + ess - hrf_delay)
                    idx_end = idx_start + stim_wind
                                        
                    if idx_end > episode_feat_len:
                        idx_end = episode_feat_len
                        idx_start = idx_end - stim_wind

                    # we get the feature vectors for multiple samples (N=stimulus_window), and flatten them
                    f = raw_feats[epi][idx_start:idx_end].flatten()
                    """
                    #idx = max(0, s + ess - hrf_delay)
                    #if idx >= episode_feat_len - hrf_delay: idx = -1
                    #f = raw_feats[epi][idx].flatten()""";
                    
                    idx_start = s + ess - hrf_delay - stim_wind
                    idx_end = idx_start + stim_wind
                    # Create a zero-padded array
                    f = np.zeros((stim_wind, raw_feats[epi].shape[1]))
                    # Compute valid range
                    valid_start = max(0, idx_start)
                    valid_end = min(idx_end, episode_feat_len)
                    # Insert valid features into the padded array
                    f[(valid_start - idx_start):(valid_end - idx_start)] = \
                         raw_feats[epi][valid_start:valid_end];
                    f=f.flatten();

                    tr_feat.append(f)

                stim_feat.append(np.concatenate(tr_feat))

        aligned_fmri[stimulus] = np.concatenate(stim_fmri).astype(np.float32)
        aligned_feat[stimulus] = np.stack(stim_feat).astype(np.float32)
        if v>=1: print(stimulus, stimulus, len(episodes), episodes[:3], \
             aligned_fmri[stimulus].shape, aligned_feat[stimulus].shape)
    
    if ret=="flat":
        aligned_fmri = np.concatenate([v for k,v in aligned_fmri.items()])
        aligned_feat = np.concatenate([v for k,v in aligned_feat.items()])

    return aligned_feat, aligned_fmri



def flat(dictionary, id_list=None, axis=0):
    if id_list is None: id_list=list(dictionary.keys())
    return np.concatenate([dictionary[l] for l in id_list], axis=axis);

def balanced_flat(dictionary, id_list=None, min_samples=None, axis=0, v=False):
    if id_list is None: id_list=list(dictionary.keys())
    if min_samples is None: 
        min_samples = np.min([dictionary[l].shape[0] for l in id_list])
    data =[]
    for l in id_list:
        n = np.round(dictionary[l].shape[0] / min_samples).astype(int)
        if n<=0: n=1;
        data.append(dictionary[l][::n])
        if v: print(l, n, end=", ")
    if v: print("");
    return np.concatenate(data, axis=axis)


"""
afeat, afmri = quickalign([lang_features], fmri, all_movie_sets, stim_wind=1, v=1)
print(flat(afeat).shape)
print(balanced_flat(afeat, v=1, min_samples=5000).shape)"""