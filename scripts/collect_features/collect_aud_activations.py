import os
import argparse
from glob import glob
import numpy as np
import torch
from torch import tensor    
import torchaudio.transforms as T
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from brainannlib.stimuli_data_loading import DecordAudioDataset
from brainannlib.ann_activations import iter_modules
from brainannlib.ann_activations import add_activation_hooks_to_layers, any_exact_match
from sklearn.preprocessing import StandardScaler


def setup_environment():
    """Set up environment variables and check CUDA availability."""
    os.environ['HF_HOME'] = "/scratch-scc/users/robert.scholz2/cache/huggingface"
    cuda_available = torch.cuda.is_available()
    import socket
    hostname = socket.gethostname()
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    import random
    seed = 43
    print("set random seed:", seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return "cuda" if cuda_available else "cpu"


def load_beats(args):
    from brainannlib.models.beats.BEATs import BEATs, BEATsConfig
    from huggingface_hub import hf_hub_download
    import torch.nn.functional as F

    ## definition and loading of the model
    #model_name = "BEATsIter3p"
    repo_id = "camenduru/beats"  
    filename = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"  # Replace with the file you want to download
    # Download the file and get the local path
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(local_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    if args.untrained:
        print("Initialize untrained model")
    else:
        model.load_state_dict(checkpoint['model'])
    model.eval(); # put the model into eval mode (i.e. not training)
    # batch_size =600;

    def collate_fn(batch):
        # cast audio snippets into float and resample them
        #resample = T.Resample(44100, 16000)
        resample = T.Resample(48000, 16000)
        prep_batch = [resample(tensor(tpl[0].mean(0)).float()) for tpl in batch]
        #print(len(batch[0]), batch[0][0].shape)
        # Find the maximum sequence length in the batch
        max_length = max(len(x) for x in prep_batch)
        #print([len(x) for x in prep_batch])
        # Pad sequences to the max length
        padded_batch = [F.pad(x, (0, max_length - len(x))) for x in prep_batch]
        padded_batch = torch.stack(padded_batch, dim=0)
        # Create a padding mask (0 for real values, 1 for padding)
        padding_mask = torch.ones(len(prep_batch), max_length, dtype=torch.uint8)
        for i, x in enumerate(prep_batch):
            padding_mask[i, :len(x)] = 0     

        return padded_batch, padding_mask.bool()
        
    md = iter_modules(model)
    layers = [m for m in md.keys() if (m.startswith("encoder.layers.") and len(m.split("."))==3)]

    #model_forwards
    # probs, outp_padding_mask = model.extract_features(inp_batch, padding_mask=inp_padding);
    return model, layers, collate_fn;


def load_whisper_small(args):
    return load_whisper(args, model_hf_path = "openai/whisper-small")

def load_whisper_largev3(args):
    return load_whisper(args, model_hf_path = "openai/whisper-large-v3")

def load_whisper(args, model_hf_path = "openai/whisper-small"):

    from transformers import AutoModel, AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
    #model_hf_path="openai/whisper-small" #model_name="whisper-small"
    #model_hf_path="openai/whisper-large-v3"
    model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True).eval()
    #model = WhisperForConditionalGeneration.from_pretrained(model_hf_path, output_hidden_states=True).eval()
    #feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)

    if args.untrained:
        print("Initialize untrained model")
        model = AutoModel.from_config(model.config).eval()

    model.eval(); # put the model into eval mode (i.e. not training)
    #batch_size =200;

    processor = WhisperProcessor.from_pretrained(model_hf_path)
    resample = T.Resample(48000, 16000)
    proc_indiv= lambda x : processor(x, return_attention_mask=True, sampling_rate=16000, return_tensors="pt")

    def collate_fn(batch):
        #print(len(batch), len(batch[0]), batch[0][0].shape);
        prep_batch = [resample(tensor(tpl[0].mean(0)).float()) for tpl in batch]
        encoded_inputs = [proc_indiv(audio) for audio in prep_batch]
        #for e in encoded_inputs: print(e['input_features'].shape)
        inp_batch = torch.concatenate([item['input_features'] for item in encoded_inputs], dim=0)
        attn_mask = torch.concatenate([item['attention_mask'] for item in encoded_inputs], dim=0)#.bool()

        return inp_batch, attn_mask 
    
    md = iter_modules(model)
    layers = [m for m in md.keys() if (m.startswith("encoder.layers.") and len(m.split("."))==3)]

    return model, layers, collate_fn;



def load_ast(args):
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    #model_name="ast-ftaudioset"
    model_hf_path="MIT/ast-finetuned-audioset-10-10-0.4593"
    model = ASTForAudioClassification.from_pretrained(model_hf_path)

    if args.untrained:
        print("Initialize untrained model")
        model = ASTForAudioClassification(model.config).eval()

    #feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path, return_attention_mask=True)
    model.eval(); # put the model into eval mode (i.e. not training)
    #batch_size =200;

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)
    #resample = T.Resample(44100, 16000)
    resample = T.Resample(48000, 16000)
    #ptf = lambda audio : feature_extractor(resample(tensor(audio[0]).float()), sampling_rate=16000).input_values[0]
    #proc_indiv= lambda x : processor(x, return_attention_mask=True, sampling_rate=16000, return_tensors="pt")
    proc_indiv= lambda x : feature_extractor(x, sampling_rate=16000, return_tensors="pt", padding=True)

    def collate_fn(batch):
        #prep_batch = [resample(tensor(x[0]).float()) for x in batch]
        prep_batch = [resample(tensor(tpl[0].mean(0)).float()) for tpl in batch]
        encoded_inputs = [proc_indiv(audio) for audio in prep_batch]
        inp_batch = torch.concatenate([item['input_values'] for item in encoded_inputs], dim=0)
        # no attention mask available
        #attn_mask = torch.concatenate([item['attention_mask'] for item in encoded_inputs], dim=0)#.bool()
        return inp_batch,0; #, [0]*len(inp_batch)
    
    # identify the layer descriptors for the model layers we want to probe
    md = iter_modules(model)
    prefix="audio_spectrogram_transformer.encoder.layer."
    #print(md.keys())
    layers = [m for m in md.keys() if (m.startswith(prefix) and len(m.split("."))==4)]

    return model, layers, collate_fn;




def collect_aud_activations(root_data_dir, model, collate_fn, layers, batch_size, device, \
            stimuli="all", n_layers=4, kept_tokens=6, context_len_s=10, args={}):
    
    """Process transcript files and extract activations."""

    actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    model.to(device)
    print("layers:", layers)
    layer_idxs = np.linspace(0, len(layers)-1, n_layers).round().astype(int)
    layers = np.array(layers)[layer_idxs].tolist()
    print("layers:", len(layers), layers)
    actv, hook_layer_dict, hooks = add_activation_hooks_to_layers(model, layers, verbose=False, comp_fn=any_exact_match)
    
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    files.sort()
    all_stimuli_dict = {f.split("/")[-1].split(".")[0]: f for f in files}
    if stimuli != "all":
        all_stimuli_dict = {k:v for k,v in all_stimuli_dict.items() if any(s in k for s in stimuli.split(','))}
    
    keys= all_stimuli_dict.keys()
    print(len(keys), list(keys)[:3], list(keys)[-3:])

    scaler = StandardScaler();
    model_name=args.model_name
    
    iterator = tqdm(enumerate(all_stimuli_dict.items()), total=len(all_stimuli_dict))
    for i, (stim_id, stim_path) in iterator:
        postfix = f"{n_layers}L{kept_tokens}T{context_len_s}S"  + ("+untr" if args.untrained else "")

        fn = f"{actv_dir}/actv-{model_name}-{stim_id}-{postfix}.npy"

        if os.path.exists(fn):
            continue
    
        # create the pytorch audio dataset & dataloader for efficient loading+preparation of batches
        #ds = DecordAudioDataset(stim_path, step_size_s=1.49, transform=None, chunk_size_s=context_len_s)
        ds = DecordAudioDataset(stim_path, step_size_s=1.49, transform=lambda x : x, chunk_size_s=context_len_s)
        stimulus_loader = DataLoader(ds, batch_size = batch_size, num_workers=0, collate_fn=collate_fn)
        red_actv={ln: [] for ln in layers}

        ###################################################################
        if model_name == "BEATs":

            for inp_batch, inp_padding in stimulus_loader:
                actv.clear()
                inp_batch=inp_batch.to(device)
                inp_padding=inp_padding.to(device)

                with torch.no_grad():
                    probs, outp_padding_mask = model.extract_features(inp_batch, padding_mask=inp_padding);

                # can be moved up as its contstant
                conv1d = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=16, stride=16, bias=False)
                conv1d.weight.data = model.patch_embedding.weight.data.mean(dim=-1).mean(dim=0, keepdim=True)

                #  1024 frequency bins in the spectrogram, subdivided into patches of grouping 16 frequencies (non-overlapping)
                num_freq_patches = (128-16) // 16 +1 

                for layer_name in layers:
                    # get the collected activations for the current batch
                    #feat_map = actv[layer_name][-1].swapaxes(0,1)
                    #feat_map[outp_padding_mask.detach().cpu().numpy()] = 0
                    #feat_map = feat_map[:,::10,:] # only collect every 10th token
                    # doesnt seem very principled
                    # e.g. result in shape (405, 11, 768) when 2s context is given
                    # or in flat: (405, 33792)

                    # better:
                    # Altogether:
                    feat_map = actv[layer_name][-1].swapaxes(0,1)
                    # extract the time-dimension conv operation from conv2d
                    
                    # corresponds to 1.49s with 9 time-patches each containing 8 frequncy patches
                    # needs to be lower or equal to the dataset stride (=TR)
                    tokens_to_extract= 72 
                    # standard BEATs forward procedure
                    fbank = model.preprocess(inp_batch, fbank_mean=15.41663, fbank_std=6.55582)
                    padding_mask = model.forward_padding_mask(fbank, inp_padding)
                    padding_mask = padding_mask.unsqueeze(1).float() 
                    # Use the 1D convolution to reduce the time-dimension
                    padding_mask = conv1d(padding_mask).squeeze().detach().cpu().numpy()
                    last_ttoken_of_interest = (padding_mask == 0).sum(1) - 1
                    start_indices = ((last_ttoken_of_interest+1) * num_freq_patches) - tokens_to_extract
                    feat_map = np.array([feat_map[i, start_idx:start_idx + tokens_to_extract] \
                                        for i, start_idx in enumerate(start_indices)])
                    #result in shape (405, 72, 768) when 2s context is of interest 
                    # i.e. 55296 dimensions (handable by pca)
                    red_actv[layer_name].append(feat_map)
                    actv[layer_name] = []


        ###################################################################
        if model_name.startswith("whisper"):

            conv1_kwargs = dict(stride=model.encoder.conv1.stride, padding=model.encoder.conv1.padding, 
                        dilation=model.encoder.conv1.dilation, groups=model.encoder.conv1.groups)
            conv2_kwargs = dict(stride=model.encoder.conv2.stride, padding=model.encoder.conv2.padding, 
                        dilation=model.encoder.conv2.dilation, groups=model.encoder.conv2.groups)

            # iterate over batches within stimulus file
            for inp_batch, attn in stimulus_loader:
                actv.clear()
                inp_batch=inp_batch.to(model.device);
                with torch.no_grad():
                    _ = model.encoder(inp_batch)
                
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

                    # for the last few tokens before padding
                    indices = out_indices.detach().cpu().numpy()
                    if indices.shape==(): # if its just one sample in the batch
                        indices = np.array([indices])
                    
                    rows = np.arange(len(feat_map))[:, None]  # Shape (n_batch, 1)
                    # last n_before indices
                    #cols = np.stack([indices - i for i in range(kept_tokens, 0, -1)], axis=1)  # Shape (n_batch, n_before)
                    
                    # Alternative: get X equally distributed tokens across the TR duration
                    # Whisper uses a 25ms window with a 10ms stride to extract mel-spectrogram frames.
                    # this is then downsampled to 20ms/token -> so 75token per 1.49s (=1TR)
                    # we exclude 0 as this is the first non-attended token (=empty/padding)
                    lags = np.linspace(75, 1, kept_tokens) 
                    cols = np.stack([indices - i for i in lags], axis=1).round().astype(int)  
                    # cols of shape (n_batch, kept_tokens)
                    
                    feat_map = feat_map[rows, cols] # shape (n_batch, kept_tokens, n_hidden)
                    #if snum < 5: print(layer_name, fshapes)
                    red_actv[layer_name].append(feat_map)
                    actv[layer_name] = []
        

        ###################################################################
        if model_name == "AST":
            for inp_batch, attn_mask in stimulus_loader:
                actv.clear()
                inp_batch=inp_batch.to(model.device);

                with torch.no_grad():
                    _ = model.audio_spectrogram_transformer(input_values=inp_batch, output_hidden_states=False)
                
                # each time patch of the patch embedding subdivides the spectrogram frequencies
                # into 12 frequency patches (hop 10, window 16 (?))
                f_patches = (128 - 16) // 10 + 1  # fixed 12 frequency patches

                # find the idx of the first non-attened patch
                first_nonatt=[]
                for j in range(len(inp_batch)):
                    arr = inp_batch[j].detach().cpu().numpy()
                    # find the most frequent mean value across mel features 
                    # this will certainly correspond to the padding
                    #unique_vals, counts = np.unique(arr.mean(1), return_counts=True)
                    # find using this most frequent mean feat value we can find a padding mask
                    #padding = arr.mean(1)==unique_vals[np.argmax(counts)]
                    # most common unique value approach doesnt work for episodes that start with silence
                    # so we hardcode that value to the known mean of the embedded padding
                    padding=arr.mean(1)==0.4670324
                    # and the first time we get a true element, this will be the onset of the padding
                    first_non_attn_mel=padding.argmax();
                    # then transform from mel to patch time resolution 
                    first_non_attn_tpatch= (first_non_attn_mel) // 10 + 1
                    # and for each time patch we have 12 frequency patches
                    first_nonatt.append(first_non_attn_tpatch* f_patches)

                first_nonatt=np.array(first_nonatt)
                # The input spectrogram has a 10ms hop size per frame
                # Each time patch moves by 10 spectogram frames (~ frame_stride), 
                # so we get 1 "time-patch" per 100ms
                # representing 160ms of audio (since it’s using a 16-mel-frame window; so there is overlaps)
                #t_patches_tr= (149) // 10 + 1 ~ round(1.49 / 0.1)
                num_patches_per_tr = round(1.49 / 0.1) * f_patches #180
                start_indices =  first_nonatt - num_patches_per_tr
                #print(num_patches_per_tr)
                #print(start_indices)
                start_indices =2+ start_indices # accounting for cls token etc

                for layer_name in layers:
                    # get the collected activations for the current batch
                    feat_map = actv[layer_name][-1]
                    #feat_map = outputs.last_hidden_state[:,:2].detach().cpu().numpy();
                    #feat_mapx = feat_map.reshape((len(feat_map),-1)) # flatten
                    #print(feat_map.shape)
                    
                    # get only the patches corresponding the most recent 1.49s (=1TR)
                    feat_map = np.array([feat_map[i, start_idx:start_idx + num_patches_per_tr] \
                        for i, start_idx in enumerate(start_indices)])
                    
                    # shape (n_samples, n_tf_patches=180, n_hidden=768)
                    # or flattened: (n_samples, 138240)
                    red_actv[layer_name].append(feat_map)
                    actv[layer_name] = []
        

        ###################################################################
        # Saving for all models

        # concatenate across batches for each layer for the last Token embeddings for the video stimulus
        #embd_dict = {layer_name: np.concatenate(red_actv[layer_name], axis=0) for layer_name in layers}
        # we dont save as pickle object anymore, because loading many files using pickle gets
        # consequtively slower, and runs into memory leaks (apparently)
        if i==0: print("After loop")
        embd_npy = []
        for layer_name in layers:
            # concat timepoints per layer
            arr = np.concatenate(red_actv[layer_name], axis=0)
            arr = arr.reshape((arr.shape[0],-1)) # flatten features per layer
            embd_npy.append(arr)

        # concatenate features along layers
        embd_npy = np.concatenate(embd_npy, axis=-1) 
        if i<=2:  print(i, embd_npy.shape)

        #if snum < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
        np.save(fn, embd_npy)

        # save also the shapes for each of the layers activations 
        # so we can reconstruct original shapes
        # using reconstruct_activations defined in untitled-1.ipynb
        if i==0:
            info_fn = f"{actv_dir}/actv-{model_name}-{postfix}.layerinfo.txt"
            info=[f"{layer_name}:{str(red_actv[layer_name][0].shape[1:])}" for layer_name in layers]
            with open(info_fn,"w") as f:
                outstring="; ".join(info)
                f.write(outstring);
            print("Saved layer info:", info_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for customization
    parser.add_argument("--stimuli", type=str, default="all", help="Stimuli set to use (e.g., 'all', 'friends_s07', 'friends_s07,friends_s06')")
    parser.add_argument("--model_name", type=str, default="whisper-small", help="Model name")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for processing (e.g., 60 for A100)")
    parser.add_argument("--kept_tokens", type=int, default=6, help="")
    parser.add_argument("--context_len_s", type=int, default=10, help="")
    parser.add_argument("--n_layers", type=int, default=4, help="")

    parser.add_argument("--untrained", action="store_true", help="Use the untrained model version", default=False)

    args = parser.parse_args()
    print(args)
    
    root_data_dir = os.environ.get("ALGONAUTS_ROOT_DIR", "./data")
    device = setup_environment()
    model_name = args.model_name;

    #if args.untrained:
    


    MODEL_LOADING_FUNCS = {"BEATs": load_beats, "whisper-small": load_whisper_small, "whisper-large-v3": load_whisper_largev3, \
                           "AST": load_ast};
    model, layers, collate_fn= MODEL_LOADING_FUNCS[model_name](args)
    kwargs=dict(stimuli=args.stimuli, n_layers=args.n_layers, kept_tokens=args.kept_tokens, \
                context_len_s=args.context_len_s, args=args)
    collect_aud_activations(root_data_dir, model, collate_fn, layers, args.batch_size, device, **kwargs)

    

    