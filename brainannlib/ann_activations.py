######################################################################################
## This file contains functions that deal with the collection, saving and loading
## of activations at the different layers of an artificial neural network
######################################################################################



from functools import partial
import collections, re, sys, os
import numpy as np
#from brainannlib.stats_and_metrics import tqdm_mem_stats, matrix_information

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
activations_path = os.path.join(root_data_dir, "ann_brain_data/activations")

import torch;

##############################################################################
# Collecting Activations



def iter_modules(curr_module, descriptor = "", mod_dict = {}, plain_style=False, rootlv=True):
  """
  Recursively iterates over all the subcomponents (i.e. layers) of a deep neural network (pytorch module)

  Args:
    descriptor (str): the current parent module descriptor string 
      e.g. "model.encoder.layers.0.ffn", default is ""

  Returns:
    mod_dict (dictionary): 
      a dictionary with the module descriptor string e.g. "model.encoder.layers.0.ffn" as keys
      and a reference/hanlde pointing to the component itself
  """
  # add the current module/layer to the dictionary
  mod_dict.update({descriptor : curr_module});

  # iterate over all the children if existing
  for child_name, module in curr_module._modules.items():
    new_descriptor = descriptor+("" if rootlv==True else ".") + child_name
    # recursively call this function to process each of the children
    _ = iter_modules(module, new_descriptor, mod_dict = mod_dict,  rootlv=False)

  return mod_dict;




# def save_activations(activations, name, print_layerinout, module, inp, out):
#   """
#   A function that saves the current activations of a given pytorch module 
#   (e.g. a neural network layer) to a dictionary for downstream usage
#   to use as a hook function for the pytorch module

#   Args:
#     activations (collections.defaultdict(list)): dictionary to save the activations to
#     name : key indexing the list to which the current activations should be appended to
#     print_layerinout : print information for debugging whenever this function is called
#     module, inp, out : default args passed to the hook function, 
#       out contains the activations we are interested in
    
#   """
#   if print_layerinout: 
#     print('Inside ' + module.__class__.__name__ + ' forward', "in:", inp[0].shape, "out:", out.shape)
  
#   # in case out contains more than one element (e.g. activations, and attentions)
#   # only return the furst
#   if isinstance(out, tuple): out = out[0]
#   activations[name].append(out.detach().cpu().float().numpy())


# # check if any 
# 
def save_activations(activations, name, print_layerinout):
    def hook(module, inp, out):
        if print_layerinout: 
            print('Inside ' + module.__class__.__name__ + ' forward', "in:", inp[0].shape, "out:", out.shape)
        
        # in case out contains more than one element (e.g. activations, and attentions)
        # only return the first
        if isinstance(out, tuple): out = out[0]
        activations[name].append(out.detach().cpu().float().numpy())
    return hook

contains_any = lambda s, incl: np.any(np.array([re.match(x, s) for x in incl])!=None) 

def any_exact_match(name, includes):
    for x in includes:
        if name==x: return True
    return False


def add_activation_hooks_to_layers(model, include_layers_containing, print_layerinout=False, comp_fn = contains_any, \
                                   print_stats = False, verbose=False):
  """
  A function to add hooks to submodules of a pytorch module (e.g. to collect layer-wise activations)

  Args:
    model (pytorch.module) : the model to which to attach the hooks
    include_layers_containing (list[string]) : a list defining to which 
      submodules/layers to attach the hooks to
    comp_fn (function) : function that compares the elements of include_layers_containing to
      the module descriptor strings dervied by iter_modules (e.g. "model.encoder.layers.0.ffn")
      could be e.g. any_exact_match which requires the list elements to perfectly 
      match the module descriptor; or contains_any which just requires the element to be 
      contained within the module descriptor

  Returns:
    actv (collections.defaultdict(list)): a reference to the dictionary to which all the 
      module/layer activations will be saved to (e.g. by future calls to the module) 
    hook_layer_dict (dict) : a mapping from module descriptors to modules
    hooks (global list) : the function creates a new global list called "hooks" that stores
      references to all the active hooks to the module; these get overwritten by each function 
      call to make sure we don't save multiple copies of the same activations.
  
  Examples: 
    md = iter_modules(model)
    # select all layer-modules that are 3 hops from the module root
    layers = [m for m in md.keys() if (m.startswith("vision_model.encoder.layers.") and len(m.split("."))==4)]
    # select only 5 layers
    layer_idxs = np.linspace(0, len(layers)-1, 5).round().astype(int)
    layers = np.array(layers)[layer_idxs].tolist()
    actv, _, hooks = add_activation_hooks_to_layers(model, layers, comp_fn = any_exact_match)
  """
  # initialize an array to store the activations in
  actv = collections.defaultdict(list)
  if not("hooks" in globals().keys()): globals()["hooks"] = []  
  for h in globals()["hooks"]: h.remove()
  md = iter_modules(model)
  hook_layer_dict = {}
  for name in md.keys():
    if not comp_fn(name, include_layers_containing): continue;
    if verbose: print("Adding hook to:", name) # end=", ");
    h1 = md[name].register_forward_hook(save_activations(actv, name, print_stats))
    hooks.append(h1)
    hook_layer_dict.update({name: md[name]})

  return actv, hook_layer_dict, globals()["hooks"]



def load_ann_embeddings(model_name, pretrained, dataset, stimuli_names, layers, 
        expected_lenghts=None, mode="eqsTR1.49s", ending="srp",v=0, use_tqdm=False, 
        actv_dir=activations_path):
    """
    Legacy function for bateched loading activations saved in ".npy" files,
    loading all the files for each of the stimuli given in stimuli_names
    retaining activations only for the layers defined in layers;
    should be streamlined
    """

    from tqdm.auto import tqdm
    if isinstance(pretrained, str):
        pt = pretrained
        
    if isinstance(pretrained, bool):
        pt = "pretr" if pretrained else "untr";

    if not (ending  ==""): ending = "-"+ending;
    if not (mode ==""): mode = "-"+mode;
    if not (pt  ==""): pt = "-"+pt;
    if not (dataset ==""): dataset = "-"+dataset;
    
    stim_embeddings = []
    iterator = tqdm(enumerate(stimuli_names), total=len(stimuli_names)) if use_tqdm else enumerate(stimuli_names);
    for i,stim_name in iterator:
        fn = actv_dir + f"/actv-{model_name}{pt}{dataset}-{stim_name}{mode}{ending}.npy"
        #print(fn)
        data = np.load(fn, allow_pickle=1).item();
        
        if not(expected_lenghts is None):
            for layer_name in layers:
                n_samples = expected_lenghts[i]
                n_data = len(data[layer_name]);
                if n_data<n_samples:
                    data[layer_name] = np.pad(data[layer_name], ((0, n_samples - n_data), (0, 0)), mode='constant')
                    print(f"Embedding for {stim_name} contains to few samples, existing {n_data} required {n_samples}. Padding by zeros to {data[layer_name].shape}")
                elif n_data>n_samples:
                    data[layer_name] = data[layer_name][:n_samples]
                    print(f"Embedding for {stim_name} contains too many samples, existing {n_data} required {n_samples}. Truncating it to {data[layer_name].shape}")
                # do padding or cropping, with zeros
        
        stim_embeddings.append(data)
        
    ann_embeddings = {}
    for layer_name in layers: 
        ann_embeddings[layer_name] = np.concatenate([emb[layer_name] for emb in stim_embeddings], axis=0)
        if v: print(layer_name, ann_embeddings[layer_name].shape)
    
    return ann_embeddings;