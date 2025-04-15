
# to make models run fast on CPUs
import os

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")

# optional; this is the path where ANN weights are stored/cached by the transformers library
os.environ['HF_HOME'] = "/scratch-scc/users/robert.scholz2/cache/huggingface"

import numpy as np
from torch import set_num_threads
from brainannlib.stats_and_metrics import get_num_assigned_cpus
set_num_threads(get_num_assigned_cpus())

from transformers import AutoModel, AutoFeatureExtractor, WhisperProcessor
import torch
import torchaudio.transforms as T
from torch import tensor
from tqdm.auto import tqdm

from brainannlib.stimuli_data_loading import DecordAudioDataset
from torch.utils.data.dataloader import DataLoader


## definition and loading of the model
model_name="whisper-small"
model_hf_path="openai/whisper-small"
n_before=4; # gather the activation for the n-last tokens
device="cuda"
#model_name="whisper-large-v3"
#model_hf_path="openai/whisper-large-v3"
model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)
model.eval(); # put the model into eval mode (i.e. not training)
model.to(device) # push the model onto the device (e.g. cuda GPU)
batch_size =200;

# initializating of two convolution layer kwargs based on the whisper-model instance
# this will help with figuring out which are the hidden states for the last-n non-padding 
# tokens later on
conv1_kwargs = dict(stride=model.encoder.conv1.stride, padding=model.encoder.conv1.padding, 
                    dilation=model.encoder.conv1.dilation, groups=model.encoder.conv1.groups)
conv2_kwargs = dict(stride=model.encoder.conv2.stride, padding=model.encoder.conv2.padding, 
                    dilation=model.encoder.conv2.dilation, groups=model.encoder.conv2.groups)

# initialize the necessesary audio preprocessing steps (i.e. to float & 16kHz + asking for attn mask)
processor = WhisperProcessor.from_pretrained(model_hf_path)
pre = lambda x : T.Resample(44100, 16000)(tensor(x[0]).float());
ptf = lambda x : processor(pre(x), return_attention_mask=True, sampling_rate=16000)#[0]#squeeze();


from lib.ann_activations import iter_modules
from lib.ann_activations import add_activation_hooks_to_layers, any_exact_match
# identify the layer descriptors for the model layers we want to probe
md = iter_modules(model, verb=False, plain_style=2)
layers = [m for m in md.keys() if (m.startswith("encoder.layers.") and len(m.split("."))==3)]
layer_idxs = np.linspace(0, len(layers)-1-1, 5).round().astype(int)
layers = np.array(layers)[layer_idxs].tolist()
print(layers)

# add the hooks to the selected model layers to collect activations
actv, hook_layer_dict, hooks = add_activation_hooks_to_layers(model, layers, verbose=False, comp_fn = any_exact_match)

from glob import glob
# find all the movie stimuli .mkv video files
files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
files.sort()
stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])


"""
## One example loop to test if this is actually working
## and try out the time and memory needed for a given batch size

file=f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
ds = DecordAudioDataset(file, step_size_s=1.49, chunk_size_s=2)
 
batch_size =200; # 25 for whisper-large but can be bigger likely

stimulus_loader = DataLoader(ds, batch_size = batch_size, num_workers=0)
stimulus_loader.dataset.transform = ptf
actv.clear()
iterator=iter(stimulus_loader)
inp_batch, attn = next(iterator)
print(inp_batch.shape, attn.shape)
inp_batch=inp_batch.to(model.device);

with torch.no_grad():
    output = model.encoder(inp_batch.squeeze()) # not keeping the output frees alot of memory
    del output

    import gc
    gc.collect()

contributing_outs = torch.nn.functional.conv1d(attn.unsqueeze(1), 
    torch.ones((1,1)+model.encoder.conv1.kernel_size).to(attn), **conv1_kwargs);
contributing_outs = torch.nn.functional.conv1d(contributing_outs, #
    torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs), **conv2_kwargs) # shape: (batchsz, 1500)

# indices for the first "non-attended" element
out_indices = np.argmax(contributing_outs == 0, axis=2).squeeze()
# e.g. for the 3rd element in the batch:
# out_indices[3], contributing_outs[3, 0, 301-4:301+1] yields:
#(tensor(301), tensor([9, 9, 8, 3, 0], dtype=torch.int32))

#layers = actv.keys();
red_actv={ln: [] for ln in layers}
for layer_name in layers:
    feat_map = actv[layer_name][-1]
    fshapes=[feat_map.shape]

    indices = out_indices.detach().cpu().numpy()
    rows = np.arange(len(feat_map))[:, None]  # Shape (n_batch, 1)
    cols = np.stack([indices - i for i in range(n_before, 0, -1)], axis=1)  # Shape (n_batch, n_before)
    feat_map = feat_map[rows, cols] # shape (n_batch, n_before, n_hidden)
    fshapes.append(feat_map.shape)
    feat_map = feat_map.reshape((len(feat_map),-1)) # flatten
    fshapes.append(feat_map.shape)
    print(layer_name, fshapes)
    red_actv[layer_name].append(feat_map)
    #actv[layer_name] = []
"""



from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# approximation for the SRP projection dimension; potentially should be higher
n_proj = 6000
srps={}

n_stim = len(list(stimuli));
# number of the last non-padding tokigns for which the embeddings to collect
n_before=4; 


# iterate across all the stimuli movie files
t = tqdm(enumerate(stimuli.items()), total= n_stim);
for snum, (stim_name, stim_path) in t:
    #snum_str = f"{snum+1}/{n_stim}"

    fn = f"{actv_dir}/actv-{model_name}-{stim_name}-2sCNKperTR-last{n_before}token.npy"
    if os.path.exists(fn): continue; 

    # create the pytorch audio dataset & dataloader for efficient loading+preparation of batches
    ds = DecordAudioDataset(stim_path, step_size_s=1.49)
    stimulus_loader = DataLoader(ds, batch_size = batch_size, num_workers=0)
    stimulus_loader.dataset.transform = ptf

    # log some info for the first 5 stimulus videos
    if snum < 5: 
        print(snum, stim_name, len(ds), stim_path)

    # arrays to store the (reduced) layer-wise activations in
    red_actv={ln: [] for ln in layers}
    srp_actv={ln: [] for ln in layers}
    
    # iterate over batches within stimulus file
    for inp_batch, attn in stimulus_loader:
        actv.clear()
        inp_batch=inp_batch.to(model.device);

        with torch.no_grad():
            output = model.encoder(inp_batch)
            del output
        
        # find the which of the internal activations correpond to non-padding tokens
        contributing_outs = torch.nn.functional.conv1d(attn.unsqueeze(1), 
            torch.ones((1,1)+model.encoder.conv1.kernel_size).to(attn), **conv1_kwargs);
        contributing_outs = torch.nn.functional.conv1d(contributing_outs, #
            torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs), **conv2_kwargs) # shape: (batchsz, 1500)
        # indices for the first "non-attended" element
        out_indices = np.argmax(contributing_outs == 0, axis=2).squeeze()

        for layer_name in layers:
            
            # get the collected activations for the current batch
            feat_map = actv[layer_name][-1]
            fshapes=[feat_map.shape]

            ####### For SRP (not recommended because its overfitting)
            feat_mapx = feat_map.reshape((len(feat_map),-1)) # flatten
            feat_mapx= scaler.fit_transform(feat_mapx)

            if not layer_name in srps.keys():
                rs = abs(hash(layer_name)) % 2**32
                srps[layer_name] = SparseRandomProjection(n_proj, random_state=rs)
                srps[layer_name].fit(feat_mapx);

            srp_actv[layer_name].append(srps[layer_name].transform(feat_mapx))
            ####### end srp
            

            ####### for the last few tokens before padding
            indices = out_indices.detach().cpu().numpy()
            if indices.shape==(): # if its just one sample in the batch
                indices = np.array([indices])
            
            rows = np.arange(len(feat_map))[:, None]  # Shape (n_batch, 1)
            # last n_before indices
            cols = np.stack([indices - i for i in range(n_before, 0, -1)], axis=1)  # Shape (n_batch, n_before)
            feat_map = feat_map[rows, cols] # shape (n_batch, n_before, n_hidden)
            fshapes.append(feat_map.shape)
            #feat_map = feat_map.reshape((len(feat_map),-1)) # flatten
            fshapes.append(feat_map.shape)
            if snum < 5: print(layer_name, fshapes)
            red_actv[layer_name].append(feat_map)
            actv[layer_name] = []
    
    #for h in hooks: h.remove()

    # concatenate across batches for each layer for the last Token embeddings for the video stimulus
    #fn = f"{actv_dir}/actv-{model_name}-{stim_name}-2sCNKperTR-last{n_before}token.npy"
    embd_dict = {layer_name: np.concatenate(red_actv[layer_name], axis=0) for layer_name in layers}
    if snum < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
    np.save(fn, embd_dict)

    # concatenate across batches for each layer for the SRP embeddings for the video stimulus
    fn = f"{actv_dir}/actv-{model_name}-{stim_name}-2sCNKperTR-sc-srp6000.npy"
    embd_dict = {layer_name: np.concatenate(srp_actv[layer_name], axis=0) for layer_name in layers}
    if snum < 5: 
        print("Saving", fn)
        print(layers[0], embd_dict[layers[0]].shape)
    np.save(fn, embd_dict)

    
    
    