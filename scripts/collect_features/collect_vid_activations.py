import os
import numpy as np
import pickle as pk
from glob import glob
from tqdm.auto import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from brainannlib.anns import load_model
from brainannlib.stimuli_data_loading import TRClipVideoDecordDataset
from brainannlib.ann_activations import iter_modules, add_activation_hooks_to_layers, any_exact_match

from sklearn.preprocessing import StandardScaler

def setup_environment():
    """Set up environment variables and check CUDA availability."""
    os.environ['HF_HOME'] = "/scratch-scc/users/robert.scholz2/cache/huggingface"
    cuda_available = torch.cuda.is_available()
    import socket
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


def load_vivit_fmri_cl(device):
    from brainannlib.models.vivit import VivitMLPContrastive

    checkpoint_dir =f"{root_data_dir}/ann_brain_data/model_weights"
    local_checkpoint_file = checkpoint_dir + "/" + "vivit-mlp_adamw_lr0.0001_decaystep_wd1e-05_bsz16_ts1_epochs10_s0.pth"
    checkpoint = torch.load(local_checkpoint_file, map_location="cpu", weights_only=False)
    model = VivitMLPContrastive(embed_dim=checkpoint['opts'].embed_dim, temperature=checkpoint['opts'].temperature)
    status = model.load_state_dict(checkpoint['model'])
    print(status)
    model.to(device)
    model.eval();
    del checkpoint;
    image_processor = model.image_processor()

    def collate_fn(batch):
        inp_batch=[];
        for tr_clip, _ in batch:
            # tr_clip shape pre: [44, 480, 720, 3], with 44 frames in a TR
            # i.e. the default decord return format: frames, height, width, rgb
            tr_clip = np.transpose(tr_clip, [0, 3, 1, 2])
            # returns format: frames, rgb, height, width e.g. [44, 3, 480, 720]
            tr_clip = tr_clip[-32:] # just select the last 32 frames of an TR, e.g. [32, 3, 480, 720]
            # If the last chunk is smaller than 32 frames, pad it with last frame
            if len(tr_clip) < 32:
                tr_clip = torch.cat([tr_clip, tr_clip[-1].unsqueeze(0).expand(32 - len(tr_clip), -1, -1, -1)])
            # preprocess the video (it needs to be a list of tensors, one for each frame)
            lst = torch.split(tr_clip, 1, 0)
            lst = [l[0] for l in lst]
            tr_clip_dict = image_processor(lst, return_tensors="pt").to(device, non_blocking=True)
            tr_clip = tr_clip_dict['pixel_values'] # tr_clip_dict has only one key: ['pixel_values'] of shape [1, 32, 3, 224, 224]
            inp_batch.append(tr_clip)

        inp_batch=torch.concatenate(inp_batch, axis=0) # shape: [batch_size, 3, 8, 256, 256]
        attn = torch.tensor([e[1] for e in batch]); # unused
        return inp_batch, attn
    
    model_forward = lambda inp_batch : model.encode_video(dict(pixel_values=inp_batch))
    return model, collate_fn, model_forward, None;




def load_slow_r50(device, args={}):

    print("loading pretrained:", not args.untrained)
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', \
                           pretrained=not args.untrained);
    model = model.eval()
    model.to(device);
    model_layer = 'blocks.5.pool'
    
    from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
    from torchvision.transforms import Compose, Lambda, CenterCrop
    transform = Compose([
        UniformTemporalSubsample(8),Lambda(lambda x: x/255.0),
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=256),
        CenterCrop(256)]
    )

    def collate_fn(batch):
        inp_batch=[];
        for tr_clip, _ in batch:
            # tr_clip shape pre: torch.Size([35, 800, 1920, 3]), with 35 samples in a TR
            # i.e. the default decord return format: frames, height, width, rgb
            tr_clip = np.transpose(tr_clip, [3, 0, 1, 2])
            # inp_batch shape trp: torch.Size([3, 5, 800, 1920])
            tr_clip = transform(tr_clip)
            # inp_batch shape post: torch.Size([3, 8, 256, 256])
            # i.e. whats typically required from Pytorch video models: [rgb, frames, height, width]
            inp_batch.append(tr_clip)

        inp_batch=torch.stack(inp_batch) # shape: [batch_size, 3, 8, 256, 256]
        attn = torch.tensor([e[1] for e in batch]); # unused
        return inp_batch, attn
    
    from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
    
    #with torch.no_grad():
    #    preds = feature_extractor(inp_batch)
    #    preds["blocks.5.pool"]
    
    #model_forward = lambda inp_batch : model(inp_batch)
    model_forward = lambda inp_batch : feature_extractor(inp_batch)["blocks.5.pool"]
    return model, collate_fn, model_forward, None;




def collect_visual_activations(root_data_dir, model_name, model, layers, collate_fn, model_forward, device, stimuli="all",\
                               reduce_by_ext_pca=False, n_layers=4, batch_size=300, target_mri_TR=1.49, args={}):

    actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    featred_dir = os.path.join(root_data_dir, "ann_brain_data/featred")

    #if not(layers is None): 
    #    layer_idxs = np.linspace(0, len(layers)-1, n_layers).round().astype(int)
    #    layers = np.array(layers)[layer_idxs].tolist()
    #    actv, hook_layer_dict, hooks = add_activation_hooks_to_layers(model, layers, verbose=False, comp_fn=any_exact_match)
    
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    files.sort()
    all_stimuli_dict = {f.split("/")[-1].split(".")[0]: f for f in files}
    if stimuli != "all":
        all_stimuli_dict = {k:v for k,v in all_stimuli_dict.items() if any(s in k for s in stimuli.split(','))}
    
    keys= all_stimuli_dict.keys()
    print(len(keys), list(keys)[:3], list(keys)[-3:])

    postfix = f"1pTR{n_layers}L" if not(layers is None) else "EMBD"
    postfix = postfix+  ("+untr" if args.untrained else "")
    scaler = StandardScaler();

    pca_fn= f"{featred_dir}/actv-{model_name}.{postfix}.all_stimuli.s7ext.pca2000.pkl"
    ext_pca =  pk.load(open(pca_fn,"rb"))[-1] if reduce_by_ext_pca else None; 
    if reduce_by_ext_pca: postfix + ".extpca"


    from tqdm.contrib.telegram import tqdm
    token="7663301481:AAHiE2BWD4bZcth-GfZ6QfWdQkPzDO1y59U"
    chat_id="765270315";
    
    iterator = tqdm(enumerate(all_stimuli_dict.items()), total=len(all_stimuli_dict), \
                    token=token, chat_id=chat_id,  \
                    desc="CollectVid "+ model_name + "."+postfix, mininterval=1)
    
    for i, (stim_id, stim_path) in iterator:
        fn = f"{actv_dir}/actv-{model_name}-{stim_id}-{postfix}.npy"
        if os.path.exists(fn):
            #print("Exists, skipping:", fn)
            continue;
        
        # create the video pytorch dataset & dataloader for efficient loading+preparation of batches
        #ds = TRSamplingDecordVDataset(stim_path, target_mri_TR, None)
        ds = TRClipVideoDecordDataset(stim_path, target_mri_TR, None)
        stimulus_loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
        
        #if layers is None:
        embd_npy = []
        for inp_batch, _ in stimulus_loader:
            inp_batch = inp_batch.to(device)
            with torch.no_grad():
                embd = model_forward(inp_batch)
                embd = embd.detach().cpu().float().numpy()
                if i==0: print(embd.shape)
                embd_npy.append(embd)
        embd_npy = np.concatenate(embd_npy, axis=0) 
        if i==0: 
            print(embd_npy.shape)
            print("save as:", fn)
        np.save(fn, embd_npy)

        """
        else # not(layers is None):
            # arrays to store the (reduced) layer-wise activations in
            embd = {ln: [] for ln in layers}
            for inp_batch, _ in stimulus_loader:
                actv.clear()
                inp_batch = inp_batch.to(device)
                with torch.no_grad():
                    embd = model_forward(inp_batch)
                
                for layer_name in layers:
                    feat_map = actv[layer_name][-1]
                    feat_mapx = feat_map.reshape((len(feat_map), -1))
                    embd[layer_name].append(feat_mapx)
                    actv[layer_name] = []
                
            #embd_dict = {layer_name: np.concatenate(embd[layer_name], axis=0) for layer_name in layers}
            #np.save(fn, embd_dict)

            if i==0: print("After loop")
            embd_npy = []
            for layer in layer_name:
                # concat timepoints per layer
                arr = np.concatenate(embd[layer_name], axis=0)
                arr = arr.reshape((arr.shape[0],-1)) # flatten features per layer
                embd_npy.append(arr)
            # concatenate features along layers
            embd_npy = np.concatenate(embd_npy, axis=-1) 
            #if snum < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
            np.save(fn, embd_npy)
        """

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Arguments for customization
    parser.add_argument("--stimuli", type=str, default="all", help="Stimuli set to use (e.g., 'all', 'friends_s07', 'friends_s07,friends_s06')")
    parser.add_argument("--model_name", type=str, default="dinov2", help="Model name")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for processing")
    parser.add_argument("--n_layers", type=int, default=4, help="")
    parser.add_argument("--reduce_by_ext_pca", action="store_true", help="Enable external PCA-based feature reduction")
    
    parser.add_argument("--untrained", action="store_true", help="Use the untrained model version", default=False)
    
    args = parser.parse_args()
    
    root_data_dir = os.environ.get("ALGONAUTS_ROOT_DIR", "./data")
    device = setup_environment()
    model_name = args.model_name;

    MODEL_LOADING_FUNCS = {"vivit-fmri-cl": load_vivit_fmri_cl, "slow_r50":load_slow_r50};
    model, collate_fn, model_forward, layers = MODEL_LOADING_FUNCS[model_name](device, args=args)

    kwargs = dict(reduce_by_ext_pca=args.reduce_by_ext_pca, n_layers=args.n_layers, batch_size=args.batch_size, \
                  target_mri_TR=1.49, stimuli=args.stimuli, args=args)
    collect_visual_activations(root_data_dir, model_name, model, layers, collate_fn, model_forward, device, **kwargs)
