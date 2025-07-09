import os
import numpy as np
import pickle as pk
from glob import glob
from tqdm.auto import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from brainannlib.anns import load_model
from brainannlib.stimuli_data_loading import TRSamplingDecordVDataset, TRClipVideoDecordDataset
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

def load_vit(args):
        #model, preprocess, model_forward = load_model(model_name, pretrained=True)   
    #     model_forward = lambda batch : model.vision_model.forward(pixel_values=batch.squeeze((1)), output_hidden_states=False);
    #     model_name="clipvit-base-patch32" #"internViT"
    #md = iter_modules(model)
    #layers = [m for m in md.keys() if (("layer" in m) and len(m.split(".")) in [1])]
    #layers = layers+["avgpool"]
    return None;


def load_Internvit300M_V2_5(args):
    from transformers import AutoModel, CLIPImageProcessor, AutoConfig
    config = AutoConfig.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5',trust_remote_code=True)
    config.use_flash_attn = False

    # HF automatically identifed flash attn as necesseary from the modeling_intern_vit file
    # evne though its not 
    import sys, importlib, types
    mock_module = types.ModuleType("flash_attn")
    mock_module.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)
    sys.modules["flash_attn"] = mock_module
    
    if not args.untrained:
        model = AutoModel.from_pretrained(
            'OpenGVLab/InternViT-300M-448px-V2_5',
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,).eval()
    else:
        print("load untrained InternViT-300M-448px-V2_5")
        model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)

    del sys.modules["flash_attn"]

    processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5')

    def transform(batch):
        images = [item[0].moveaxis(-1, 0) for item in batch]
        targets = [item[1] for item in batch]
        pix_vals = processor(images=images, return_tensors="pt")["pixel_values"]
        return pix_vals, targets

    model_forward = lambda batch : model(batch)
    
    layers = ["encoder"]
    return model, layers, transform, model_forward;
    


def load_clip_vit(args):
    from transformers import CLIPProcessor, CLIPModel, CLIPConfig
    if not args.untrained:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        print("load untrained CLIPViT")
        cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel(cfg)
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #preprocess = lambda frame: processor(images=frame, return_tensors="pt", padding=True)["pixel_values"]
    #preprocess = lambda frame: processor(images=frame.moveaxis(-1,0), return_tensors="pt", padding=True)["pixel_values"]
    #include_layers_containing = ["vision_model.*[0-9]+.mlp.fc2"] #["fc2"] # "vision_model.*(0|4|8|11).mlp.fc2"

    def transform(batch):
        images = [item[0].moveaxis(-1, 0) for item in batch]
        targets = [item[1] for item in batch]
        pix_vals = processor(images=images, return_tensors="pt", padding=True)["pixel_values"]
        return pix_vals, targets

    #model_desc = "CLIPb32"
    model_forward = lambda batch : \
       model.get_image_features(pixel_values = batch.squeeze((1)))
    
    md = iter_modules(model)
    layers = [m for m in md.keys() if (m.startswith("vision_model.encoder") and len(m.split("."))==5)]
    #return model, preprocess, model_forward
    return model, layers, transform, model_forward;


def load_dinov2(args):
    from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
    model_hfid = "facebook/dinov2-large-imagenet1k-1-layer"
    processor = AutoImageProcessor.from_pretrained(model_hfid)

    if args.untrained:
        config = AutoConfig.from_pretrained(model_hfid)
        model = AutoModelForImageClassification.from_config(config)
    else:
        model = AutoModelForImageClassification.from_pretrained(model_hfid)
    model.eval()

    def transform(batch):
        images = [item[0].moveaxis(-1, 0) for item in batch]
        targets = [item[1] for item in batch]
        return processor(images=images, return_tensors="pt"), targets

    def model_forward(inputs):
        return model.dinov2(**inputs, output_hidden_states=False)

    md = iter_modules(model)
    layers = [m for m in md.keys() if (("encoder.layer." in m) and len(m.split("."))==4)]
    return model, layers, transform, model_forward

def load_resnet50(args):
    from torchvision.models import resnet50, ResNet50_Weights
    import torchvision.transforms as T
    weights= ResNet50_Weights.DEFAULT
    model = resnet50(weights= weights) if not args.untrained else resnet50();
    model.eval()
    preprocess = weights.transforms()

    def transform(batch):
        images = [preprocess(T.ToPILImage()(tuple[0].moveaxis(-1, 0))) for tuple in batch]
        targets = [tuple[1] for tuple in batch]
        return torch.stack(images), targets

    model_forward = lambda x : model(x);

    md = iter_modules(model)
    layers = [m for m in md.keys() if (("layer" in m) and len(m.split(".")) in [1])]
    print(layers)

    return model, layers, transform, model_forward



def collect_visual_activations(root_data_dir, model_name, model, layers, transform, model_forward, device, stimuli="all",\
                               reduce_by_ext_pca=False, n_layers=4, batch_size=300, target_mri_TR=1.49, args={}):

    actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    featred_dir = os.path.join(root_data_dir, "ann_brain_data/featred")

    model.to(device)
    layer_idxs = np.linspace(0, len(layers)-1, n_layers).round().astype(int)
    layers = np.array(layers)[layer_idxs].tolist()
    actv, hook_layer_dict, hooks = add_activation_hooks_to_layers(model, layers, verbose=False, comp_fn=any_exact_match)
    
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    files.sort()
    all_stimuli_dict = {f.split("/")[-1].split(".")[0]: f for f in files}
    if stimuli != "all":
        all_stimuli_dict = {k:v for k,v in all_stimuli_dict.items() if any(s in k for s in stimuli.split(','))}
    
    keys= all_stimuli_dict.keys()
    print(len(keys), list(keys)[:3], list(keys)[-3:])
    
    postfix = f"{args.imgs_per_tr}pTR{n_layers}L" + ("+untr" if args.untrained else "")
    scaler = StandardScaler();

    pca_fn= f"{featred_dir}/actv-{model_name}.{postfix}.all_stimuli.s7ext.pca2000.pkl"
    ext_pca =  pk.load(open(pca_fn,"rb"))[-1] if reduce_by_ext_pca else None; 
    if reduce_by_ext_pca: postfix + ".extpca"
    
    from tqdm.contrib.telegram import tqdm
    token="7663301481:AAHiE2BWD4bZcth-GfZ6QfWdQkPzDO1y59U"
    chat_id="765270315";
    
    iterator = tqdm(enumerate(all_stimuli_dict.items()), total=len(all_stimuli_dict), \
                    token=token, chat_id=chat_id,  \
                    desc="CollectVis "+ model_name + "."+postfix, mininterval=1)
    
    for i, (stim_id, stim_path) in iterator:
        fn = f"{actv_dir}/actv-{model_name}-{stim_id}-{postfix}.npy"
        if os.path.exists(fn):
            continue
            print("Exists,", fn)
            #print("overwrite", fn)
        
        # create the video pytorch dataset & dataloader for efficient loading+preparation of batches
        #ds = TRSamplingDecordVDataset(stim_path, target_mri_TR, None)
        # the new dataset will have multiple images per item
        ds = TRClipVideoDecordDataset(stim_path, target_mri_TR, None)

        def outer_collate(batch):
            # this gets a batch of minibatches (i.e. short clips of TR lengths)
            num_elements=args.imgs_per_tr;
            
            
            batches = []
            for b in range(len(batch)):
                minibatch = batch[b][0]
                if i==0 and b==0: print(b, minibatch.shape)
                indices = torch.linspace(0, len(minibatch) - 1, num_elements).long()
                minibatch = minibatch[indices];
                minibatch = [(m, 0) for m in minibatch] # make it tuples so the collate can work
                minibatch = transform(minibatch)[0] # preprocess
                #minibatch = torch.stack(minibatch)
                if i==0 and b==0: print("minib:", minibatch.shape)
                batches.append(minibatch)

            targets = [0] * len(batches);
            batch=torch.concatenate(batches, axis=0);
            

            #if i==0 and b==0:print("Final:", batch.shape) # [batch_size, num_elements, 3, 224, 224] for resnet50
            """
            # reconstruct
            num_elements =8
            n_trs = feat_map.shape[0] // num_elements
            original_shape = (n_trs, num_elements) + feat_map.shape[1:]
            feat_map = feat_map.reshape(original_shape)
            """;
            return batch, torch.tensor(targets);

        #stimulus_loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=transform)
        stimulus_loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=outer_collate)
        
        
        # arrays to store the (reduced) layer-wise activations in
        embd = {ln: [] for ln in layers}
        for inp_batch, _ in stimulus_loader:
            actv.clear()
            if i==0: print(inp_batch.shape)
            if model_name== "Internvit300M_V2_5": inp_batch=inp_batch.bfloat16()
            inp_batch = inp_batch.to(device)
            with torch.no_grad():
                _ = model_forward(inp_batch)
            
            for layer_name in layers:
                feat_map = actv[layer_name][-1]
                feat_mapx = feat_map.reshape((len(feat_map), -1))
                
                embd[layer_name].append(feat_mapx)
                actv[layer_name] = []
                
        #embd_dict = {layer_name: np.concatenate(embd[layer_name], axis=0) for layer_name in layers}
        #np.save(fn, embd_dict)

        if i==0: print("After loop")
        embd_npy = []
        for layer_name in layers:
            # concat timepoints per layer
            arr = np.concatenate(embd[layer_name], axis=0)
            if i==0: print(arr.shape)
            n_trs = arr.shape[0] // args.imgs_per_tr
            original_shape = (n_trs, args.imgs_per_tr) + feat_map.shape[1:]
            arr = arr.reshape(original_shape)
            if i==0: print(arr.shape)

            if model_name == "Internvit300M_V2_5":
                embedding_cls = arr[:,:,0,:]
                embedding_non_cls = arr[:,:,1:,:]
                avg_embedding_non_cls = np.mean(embedding_non_cls, axis=2)
                arr = np.stack([embedding_cls, avg_embedding_non_cls], axis=2)
            else:
                arr = arr.reshape((arr.shape[0],-1)) # flatten features per layer
            if i==0: print(arr.shape)
            embd_npy.append(arr)
        # concatenate features along layers
        embd_npy = np.concatenate(embd_npy, axis=-1) 
        if i==0: print(embd_npy.shape)

        if not reduce_by_ext_pca:
            #if snum < 5: print("Saving", fn ,layers[0], embd_dict[layers[0]].shape)
            np.save(fn, embd_npy)
            # save also the shapes for each of the layers activations 
            # so we can reconstruct original shapes
            # using reconstruct_activations defined in untitled-1.ipynb
            if i==0:
                info_fn = f"{actv_dir}/actv-{model_name}-{postfix}.layerinfo.txt"
                info=[f"{layer_name}:{str(embd[layer_name][0].shape[1:])}" for layer_name in layers]
                with open(info_fn,"w") as f:
                    outstring="; ".join(info)
                    f.write(outstring);
                print("Saved layer info:", info_fn)
        
        else:
            embd_npy = scaler.fit_transform(embd_npy)
            embd_npy = ext_pca.transform(embd_npy)
            if i==0: print("after reduction:", embd_npy.shape)
            np.save(fn, embd_npy)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Arguments for customization
    parser.add_argument("--stimuli", type=str, default="all", help="Stimuli set to use (e.g., 'all', 'friends_s07', 'friends_s07,friends_s06')")
    parser.add_argument("--model_name", type=str, default="dinov2", help="Model name")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for processing")
    parser.add_argument("--n_layers", type=int, default=4, help="")
    parser.add_argument("--reduce_by_ext_pca", action="store_true", help="Enable external PCA-based feature reduction")
    
    # to be implemented
    parser.add_argument("--imgs_per_tr", type=int, default=1, help="")
    parser.add_argument("--untrained", action="store_true", help="Use the untrained model version", default=False)

    args = parser.parse_args()
    
    root_data_dir = os.environ.get("ALGONAUTS_ROOT_DIR", "./data")
    device = setup_environment()
    model_name = args.model_name;

    MODEL_LOADING_FUNCS = {"dinov2": load_dinov2, "resnet50": load_resnet50, "CLIPViT": load_clip_vit,
                           "Internvit300M_V2_5": load_Internvit300M_V2_5};
    
    model, layers, transform, model_forward = MODEL_LOADING_FUNCS[model_name](args)

    kwargs = dict(reduce_by_ext_pca=args.reduce_by_ext_pca, n_layers=args.n_layers, batch_size=args.batch_size, \
                  target_mri_TR=1.49, stimuli=args.stimuli, args=args)
    collect_visual_activations(root_data_dir, model_name, model, layers, transform, model_forward, device, **kwargs)
