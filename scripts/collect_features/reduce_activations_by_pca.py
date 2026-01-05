import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from glob import glob
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from brainannlib.feature_loading import quickalign, load_and_concat_activations
from brainannlib.stats_and_metrics import corr_score, get_joint_explained_variance
from brainannlib.monitoring import list_biggest_vars
import gc

def setup_environment():
    """Set up environment"""
    import socket
    hostname = socket.gethostname()
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())


def load_and_concat_activations(file, layers=None, swapaxes=True, scale_before_flat=False):
    data = np.load(file, allow_pickle=True)
    if data.shape == (): 
        #print("is dict")
        item = data.item()  # unpack dict
        del data; gc.collect()
        data= item;

    if isinstance(data, dict):
        if layers is None:
            layers = list(data.keys())

        for l in layers:
            if l not in data:
                print(f"Warning: missing layer {l} in file {file}")

        first_shape = data[layers[0]].shape[1:]
        different_layer_dims = any(data[l].shape[1:] != first_shape for l in layers)

        if different_layer_dims:
            data_list = []
            for l in layers:
                arr = data[l]
                if scale_before_flat:
                    arr = StandardScaler().fit_transform(arr.reshape((arr.shape[0], -1))).reshape(arr.shape)
                arr = arr.reshape((arr.shape[0], -1))
                data_list.append(arr)
                del arr
            data = np.concatenate(data_list, axis=-1).astype(np.float32)
            del data_list
            gc.collect()
            swapaxes = False
        else:
            data = np.stack([data[l] for l in layers]).astype(np.float32)
            gc.collect()

    orig_shp = data.shape
    if swapaxes:
        data = np.swapaxes(data, 0, 1)  # layer, time, token, hidden → time, layer, token, hidden

    data = data.astype(np.float32)
    return data, orig_shp, layers


def load_features_for_stimuli(root_data_dir, model_name, postfix, stimuli, swapaxes=True, args={}):
    actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    mvid = lambda x : x.split("/")[-1].split("-")[2+model_name.count("-")]

    tmpl = f"{actv_dir}/*{model_name}*{postfix}.npy"
    all_esti_files = glob(tmpl)
    print("## Loading ANN-feature data for PCA estimation: ")
    print("template: ", tmpl)
    print("Found files: ", len(all_esti_files))
    
    if stimuli!="all":
        all_esti_files = [f for f in all_esti_files if  any(s in f for s in stimuli.split(','))]
        print("Filters:", stimuli)
        print("Filtered files: ", len(all_esti_files))
        print(all_esti_files)
    print(len(all_esti_files), list(all_esti_files)[:3], list(all_esti_files)[-3:])

    scaler = StandardScaler()

    # Alternative
    train_features = []; layers=None; stride=args.est_on_every_n;
    for i, stim_activations_file in tqdm(enumerate(all_esti_files), total=len(all_esti_files)):
        data, shp, layers=load_and_concat_activations(stim_activations_file, layers=layers, swapaxes=swapaxes);
        if args.slice != "":
            data = data[args.slice]
            shp = str(shp) + ">>" + str(data.shape)
        # data e.g of shape (514, 5, 6, 2048) or (514, 100224)
        #if i==0: print(mvid(stim_activations_file), shp, "-->", data.shape) 
        print(mvid(stim_activations_file), shp, "-->", data.shape) 
        reshaped = data.reshape((data.shape[0],-1))
        del data; gc.collect()
        sc = StandardScaler()
        train_features.append(sc.fit_transform(reshaped)[::stride])
        del sc
        del reshaped; gc.collect()
            
    train_features = np.concatenate(train_features)
    gc.collect()
    return train_features


def fit_pca(root_data_dir, model_name, postfix, red_train_features, n_components=2000, args={}):

    featred_dir = f"{root_data_dir}/ann_brain_data/featred"

    newpostfix = postfix if args.newpostfix =="" else args.newpostfix

    fn= f"{featred_dir}/actv-{model_name}.{newpostfix}.all_stimuli.s7ext.pca2000.pkl"
    if os.path.exists(fn):
        print("Skipping, as it already exists:", fn)
        return;

    # Reduce the features to 2000 
    pca = PCA(n_components=args.n, random_state=1004, svd_solver="randomized")    
    # here make sure reduction is feasible by selecting approx 5k samples

    print("Estimating PCA on:", red_train_features.shape, "...")
    #red_train_features = scaler.fit_transform(red_train_features)
    red_train_features = pca.fit_transform(red_train_features)
    print("Var expl:", pca.explained_variance_ratio_.sum().round(3) ) # with 2000 comps: 0.947
    # up till here 15s; with 2000 components: 2min30s
    #feat_reduction = make_pipeline(scaler, pca)
    feat_reduction = make_pipeline(pca)

    # Save the estimated feature reduction pipeline
    print("Saving PCA: ", fn)
    pk.dump(feat_reduction, open(fn,"wb"))
    """
    # Check for how much variance in the full train_features the components account for
    embd_feat = feat_reduction.transform(train_features)  # 20s
    scaled_truth= scaler.transform(train_features) # 16s
    _,_, ratio = get_joint_explained_variance(embd_feat, pca, scaled_truth) # 23s
    print("variance explained in full train_features:", ratio.round(3)) 
    # stats: 
    # estimated on (23472, 10240) train_features[::10] ~ 0.618, on [::5] ~ 0.74, and on 
    # estimated on (23472, 10240) train_features[::2] ~ 0.933
    # reference: PCA decomposition based on friends season 1-5, [::10] ~ 0.995
    del train_features  """;
    del red_train_features
    return feat_reduction;


def reduce_stimuli_to_single_file(root_data_dir, model_name, postfix, stimuli, just_compile, swapaxes=True, args={}):
    
    actv_dir = os.path.join(root_data_dir, "ann_brain_data/activations")
    featred_dir = f"{root_data_dir}/ann_brain_data/featred"
    mvid = lambda x : x.split("/")[-1].split("-")[2+model_name.count("-")]

    tmpl = f"{actv_dir}/*{model_name}*{postfix}.npy"
    all_red_files = glob(tmpl)
    print("## Reducing ANN-feature data: ")
    print("template: ", tmpl)
    print("Found files: ", len(all_red_files))

    newpostfix = postfix if args.newpostfix == "" else args.newpostfix

    if stimuli!="all":
        all_red_files = [f for f in all_red_files if  any(s in f for s in stimuli.split(','))]
        print("Filters:", stimuli)
        for f in all_red_files: print(f)
    print(len(all_red_files), list(all_red_files)[:3], list(all_red_files)[-3:])
    
    # loading reference feature_reduction pipeline
    if not just_compile:
        fn= f"{featred_dir}/actv-{model_name}.{newpostfix}.all_stimuli.s7ext.pca2000.pkl"
        print("Using PCA: ", fn)
        feat_reduction =pk.load(open(fn,"rb"))
        scaler = StandardScaler()

    features = {} # will contain ~ 5.5GB
    var_explained= {}
    layers =  None # will be set after the first pass (in case of a dict), 
    # to make sure its the same
    
    red_fn= f"{featred_dir}/actv-{model_name}.{newpostfix}.all_stimuli.s7ext.pca2000.npy"
    var_fn= f"{featred_dir}/actv-{model_name}.{newpostfix}.all_stimuli.s7ext.pca2000.varexpl.npy"

    if os.path.exists(red_fn):
        print("Loading existing ", red_fn)
        features = np.load(red_fn, allow_pickle=1).item();
        print("Loading existing ", var_fn)
        var_explained = np.load(var_fn, allow_pickle=1).item(); 

    for i, stim_activations_file in tqdm(enumerate(all_red_files), total=len(all_red_files)):
        epi_name= mvid(stim_activations_file);
        if epi_name in features.keys(): continue;
        #print(stim_activations_file)

        data, shp, layers=load_and_concat_activations(stim_activations_file, layers=layers, swapaxes=swapaxes);
        if args.slice != "":
            data = data[args.slice]
            shp = str(shp) + ">>" + str(data.shape)
        #if len(data.shape)>3: data= data[:,:,-1,:]
        data = data.reshape((data.shape[0],-1))
        # independently scale runs first
        data_tf = data.copy() if just_compile else feat_reduction[-1].transform(scaler.fit_transform(data))       

        #data_tf=feat_reduction.transform(data)
        if i==0:  print(epi_name, shp, "-->", data.shape, "-->", data_tf.shape)#
        features[epi_name] = data_tf.astype(np.float32)

        if not just_compile:
            # check how much variance is beeing explained by the reference components
            scaled_truth= scaler.fit_transform(data) # 16s
            var_expl, var_full, ratio = get_joint_explained_variance(data_tf, feat_reduction[-1], scaled_truth)
            var_explained[epi_name] = [var_expl, var_full, ratio]
        
        del data
        if i%50==0: gc.collect();
            

    # Saving them for faster loading next time
    print("Saving compiled features:", red_fn)
    print(len(features.keys()), features["friends_s02e01a"].shape);
    np.save(red_fn, features)
    
    if not just_compile:
        print("Saving var-expl-info:", var_fn)
        np.save(var_fn, var_explained)

        # Check how much variance of the full activations is 
        # explained by the partial activations
        print("\n## Var-Expl Summary:")
        e = np.array([v[0] for k,v in var_explained.items()])
        f = np.array([v[1] for k,v in var_explained.items()])
        clip_ratios = [e[i,:].sum()/f[i,:].sum() for i in range(len(e))];
        ratio=e[:,:].sum() / f[:,:].sum()
        print(ratio.round(3), "\t", len(clip_ratios), np.array(clip_ratios).round(3)[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for customization
    parser.add_argument("--fit_pca", action="store_true", help="")
    parser.add_argument("--est_stimuli", type=str, default="all", help="Stimuli set to use (e.g., 'all', 'friends_s07', 'friends_s07,friends_s06')")
    parser.add_argument("--model_name", type=str, default="whisper-small", help="Model name")
    parser.add_argument("--postfix", type=str, default="4L", help="Postfix defining #layers etc")
    
    parser.add_argument("--reduce_stimuli", action="store_true", help="Directly reduce the stimuli by (pre-)estimated PCA-based feature reduction,"+ \
                        "unless --just_compile is specified, in which case the indiviudal files are just loaded and concatenated (assuming e.g. features have been already pre-reduced)")
    parser.add_argument("--red_stimuli", type=str, default="all", help="Stimuli set to use (e.g., 'all', 'friends_s07', 'friends_s07,friends_s06')")
    
    parser.add_argument("--just_compile", action="store_true", help="")
    parser.add_argument("--noswapaxes", action="store_true", default=False, help="Swap first two dimensions for individual stimuli feature files.")
    parser.add_argument("--n", type=int, default=2000, help="number of pca components")

    parser.add_argument("--est_on_every_n", type=int, default=5, help="number of pca components")

    # to be implemented
    parser.add_argument("--slice", type=str, default="", help="which part of the loaded activations to use, defaults to all")
    # slice works at end of load_and_concat_activations (?)
    parser.add_argument("--newpostfix", type=str, default="", help="changing the postfix for the outputs")


    args = parser.parse_args()
    print("Arguments:", args)
    
    root_data_dir = os.environ.get("ALGONAUTS_ROOT_DIR", "./data")
    setup_environment()

    if args.slice !="":
        # e.g. "(slice(None), slice(None), slice(-6, None), slice(None))"
        # or "(:,:,slice(-6, None))"
        slice_str = args.slice.replace(":", "slice(None)")
        args.slice = eval(slice_str)

    if args.fit_pca:
        train_features = load_features_for_stimuli(root_data_dir,args.model_name, args.postfix, 
                                                   args.est_stimuli, not args.noswapaxes, args=args)

        print(train_features.shape) # (23472, 10240)
        _ = fit_pca(root_data_dir, args.model_name, args.postfix, train_features, args=args)

    if args.reduce_stimuli:
        reduce_stimuli_to_single_file(root_data_dir, args.model_name, args.postfix, args.red_stimuli, 
                                      args.just_compile, not args.noswapaxes, args=args)
