######################################################################################
## This is an example file on how to collect the activations for visual transformers
## such as the CLIP-visual encoder or InternViT
## other visual networks likely need a slightly different approach (i.e. a change to 
## the probed layers and the definition of which activations to extract)
######################################################################################

import os
import numpy as np

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
actv_dir = os.environ["ALGONAUTS_ACTIVATIONS_DIR"]
#os.path.join(root_data_dir, "ann_brain_data/activations")

# optional; this is the path where ANN weights are stored/cached by the transformers library
#os.environ['HF_HOME'] = "/home/bagga005/algo/comp_data/hf"

from brainannlib.anns import load_model
from torch.utils.data.dataloader import DataLoader
from brainannlib.stimuli_data_loading import TRSamplingDecordVDataset
from brainannlib.ann_activations import iter_modules
from brainannlib.ann_activations import add_activation_hooks_to_layers, any_exact_match

from tqdm.auto import tqdm
from glob import glob

import torch
# check if cuda is available

print(torch.cuda.is_available())


# Setting the variables
model_name="clipvit-base-patch32" #"internViT"
device="cuda"
max_n_layers = 5;
target_mri_TR = 1.49 #s
batch_size = 200


# Collecting the paths to all the movie stimuli
files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
files.sort()
stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])


# loading the model
model, preprocess, model_forward = load_model(model_name, pretrained=True)   
model.eval()
model.to(device)
model_forward = lambda batch : model.vision_model.forward(pixel_values=batch.squeeze((1)), output_hidden_states=False);

# identify the layer descriptors for the model layers we want to probe
md = iter_modules(model)
layers = [m for m in md.keys() if (m.startswith("vision_model.encoder.layers.") and len(m.split("."))==4)]
# select select max_n_layers equally distributed layers
layer_idxs = np.linspace(0, len(layers)-1, max_n_layers).round().astype(int)
layers = np.array(layers)[layer_idxs].tolist()
print(layers)
# add the hooks to the selected model layers to collect activations
actv, hook_layer_dict, hooks = add_activation_hooks_to_layers(model, layers, verbose=False, comp_fn = any_exact_match)

def color_print(*args, color=None):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m'  # Added white
    }
    # Use white if color is None or not in the dictionary
    color_code = colors.get(color, colors['white'])
    # Convert all arguments to strings and join them with spaces
    text = ' '.join(str(arg) for arg in args)
    return print(f"{color_code}{text}\033[0m")

"""

## One example loop to test if this is actually working
## and try out the time and memory needed for a given batch size

stim_path= root_data_dir + "/algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
ds = TRSamplingDecordVDataset(stim_path, target_mri_TR, None)

batch_size = 200
stimulus_loader = DataLoader(ds, batch_size = batch_size, num_workers=0)
stimulus_loader.dataset.transform = preprocess

inp_batch, labels = next(iter(stimulus_loader))
inp_batch= inp_batch.to(device)

with torch.no_grad():       
  #output = model.get_image_features(pixel_values = inp_batch.squeeze((1)))
  output = model_forward(inp_batch) 

actv.clear();
"""

from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# approximation for the SRP projection dimension; potentially should be higher
n_proj = 6000   
srps={} # dict to cache layer-speciifc SRPs

# iterate across all the stimuli movie files
iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
for i, (stim_id, stim_path) in iterator:

    fn = f"{actv_dir}/actv-{model_name}-{stim_id}-eqsTR1.49s-{max_n_layers}layers-srp.npy"
    if os.path.exists(fn): continue; 
    
    # create the video pytorch dataset & dataloader for efficient loading+preparation of batches
    ds = TRSamplingDecordVDataset(stim_path, target_mri_TR, None)
    stimulus_loader = DataLoader(ds, batch_size = batch_size, num_workers=0)
    stimulus_loader.dataset.transform = preprocess

    # log some info for the first 5 stimulus videos
    color_print(i, stim_id, len(ds), stim_path, color='green')
    
    # arrays to store the (reduced) layer-wise activations in
    cls_actv={ln: [] for ln in layers}
    srp_actv={ln: [] for ln in layers}
    full_actv={ln: [] for ln in layers}

    # iterate over batches within stimulus file
    for inp_batch, labels in stimulus_loader:
        actv.clear()
        inp_batch=inp_batch.to(model.device);

        with torch.no_grad():
            output = model_forward(inp_batch)#.squeeze())
            #proj=model.visual_projection(output[1])
            #del output
        
        for layer_name in layers:
            color_print(layer_name, color='red')
            # get the collected activations for the current batch
            feat_map = actv[layer_name][-1]
            fshapes=[feat_map.shape]
            print('actv len', len(actv))
            print('actv[layer_name] len', len(actv[layer_name]))
            print('feat_map', feat_map.shape)
            print('output[1]', output[1].shape)
            ####### For CLS (pooled) activations
            # just appaned the embedding for the "first" token, i.e.
            # the one that is supposed to accumulate information 
            # across all image patches
            #cls_actv[layer_name].append(feat_map[:, 0, :]) #requires also additional post_layer_norm            cls_actv[layer_name].append(pooled)
            pooled = output[1].detach().cpu().float().numpy()
            cls_actv[layer_name].append(pooled);
            full_actv[layer_name].append(feat_map);
            fshapes.append(cls_actv[layer_name][-1].shape)
            # drawback: CLS token may only capute most of the relevant information in the last layer
            # as there is no garantuee it similiarly pools information at the other prev. layers

            ####### For SRP (not recommended because its overfitting)
            feat_mapx = feat_map.reshape((len(feat_map),-1)) # flatten
            feat_mapx= scaler.fit_transform(feat_mapx) # standartize
            if not layer_name in srps.keys():
                # use the same SRP for a given layer, so that features correspond to each
                # other across stimuli; also only cache the SRPs as their initialization
                # is slow; to allow for reproduceability, use the hash of the layer name
                # as random seed
                print(f"Creating SRP for layer {layer_name} to project from {feat_mapx.shape} to {n_proj} dims")
                rs = abs(hash(layer_name)) % 2**32
                srps[layer_name] = SparseRandomProjection(n_proj, random_state=rs)
                srps[layer_name].fit(feat_mapx);

            # transform the flattened activations using the SRP and append
            srp_actv[layer_name].append(srps[layer_name].transform(feat_mapx))
            fshapes.append(srp_actv[layer_name][-1].shape)
            ####### end SRP#
            if i < 5: print(layer_name, fshapes)
            actv[layer_name] = [] # free memory

        del output

    # concatenate across batches for each layer for the CLS embeddings for the video stimulus
    fn = f"{actv_dir}/actv-{model_name}-{stim_id}-eqsTR1.49s-{max_n_layers}layers_clsEmbd.npy"
    embd_dict = {layer_name: np.concatenate(cls_actv[layer_name], axis=0) \
                 for layer_name in layers}
    if i < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
    np.save(fn, embd_dict) # save file

    fn = f"{actv_dir}/actv-{model_name}-{stim_id}-eqsTR1.49s-{max_n_layers}layers_fullEmbd.npy"
    embd_dict = {layer_name: np.concatenate(full_actv[layer_name], axis=0) \
                 for layer_name in layers}
    if i < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
    np.save(fn, embd_dict)
    
    fn = f"{actv_dir}/actv-{model_name}-{stim_id}-eqsTR1.49s-{max_n_layers}layers-srp.npy"
    # concatenate across batches for each layer for the SRP embeddings for the video stimulus
    embd_dict = {layer_name: np.concatenate(srp_actv[layer_name], axis=0) \
                 for layer_name in layers}
    if i < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
    np.save(fn, embd_dict) # save file

    
