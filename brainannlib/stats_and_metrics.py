


import numpy as np
import os, psutil




##############################################################################
## Scores etc

def corr_score(Y_test, Y_pred):
  corr = lambda ytest, ypred : np.corrcoef(ytest, ypred)[0,1]
  return np.array([corr(Y_test[:,i], Y_pred[:,i]) for i in range(Y_pred.shape[1])])


def simple_corr(A,B):
    # Compute column-wise correlation
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    A_std = A.std(axis=0, ddof=1)
    B_std = B.std(axis=0, ddof=1)
    correlations = ((A - A_mean) * (B - B_mean)).mean(axis=0) / (A_std * B_std)
    return correlations



from sklearn.decomposition import PCA
def matrix_information(m, do_sum=True, method="pca"):
  if m.shape[1] > 1000000:
    #print("Too big", m.shape, end="...")
    return np.nan
  pca = PCA(n_components=np.min(m.shape))
  pca.fit(m)
  return pca.explained_variance_.sum()


def get_explained_variance(embd, pca, full, n_comps=None, v=False, use_tqdm=False):
    from tqdm.auto import tqdm
    n_components= embd.shape[1]
    var_full = np.linalg.norm(full - pca.mean_)
    print(var_full)
    var_embd = np.zeros(n_components)
    var_ratio = np.zeros(n_components)
    
    result = np.zeros(n_components)
    
    if n_comps is None:
        n_comps = n_components;
        
    prange = tqdm(range(n_comps)) if use_tqdm else range(n_comps);
    for ii in prange:
        X_trans_ii = np.zeros_like(embd)
        X_trans_ii[:, ii] = embd[:, ii]
        X_approx_ii = pca.inverse_transform(X_trans_ii)
        
        var_embd[ii] = np.linalg.norm(X_approx_ii - full)
        var_ratio[ii] = 1 - (var_embd[ii] / var_full) ** 2
    return var_embd, var_full, var_ratio

import numpy as np

def get_joint_explained_variance(embd, pca, full, n_comps=None):
    if n_comps is None: n_comps = embd.shape[1]
    # Center the full matrix (if not already centered)
    centered_full = full - pca.mean_
    var_full = np.var(centered_full, axis=0)
    var_full_sum = np.sum(var_full)

    # Project the data onto the first n_comps components
    X_trans_joint= np.zeros_like(embd)
    X_trans_joint[:, :n_comps] = embd[:, :n_comps]
    
    # Reconstruct the data using the first n_comps components
    full_approx = X_trans_joint.dot(pca.components_)
    
    # Calculate the reconstruction error (unexplained variance)
    difference  =centered_full -full_approx
    unexplained_variance = np.var(difference, axis=0)  
    unexplained_variance_sum = unexplained_variance.sum()    
    # The explained variance is the total variance minus the unexplained variance
    explained_variance_sum = var_full_sum - unexplained_variance_sum
    explained_variance = var_full - unexplained_variance;
    #print(explained_variance_sum, explained_variance.sum())
    # Explained variance ratio
    var_ratio = explained_variance_sum / var_full_sum
    return explained_variance, var_full, var_ratio


def make_df_(sdict, subj="unknown", layer_names=None):
    import pandas as pd
    df = pd.DataFrame(sdict["scores"])
    df['layer'] = range(len(df))
    df['subj'] = subj
    if not (layer_names is None): df['layer'] = layer_names
    return df.melt( id_vars=['layer','subj'], var_name='comp', value_name='r')


def average_scores_across_subjects(model_name, dataset, subjects, pt="pretr", score_type="fullpca", do_df=True):
    from lib.config import scores_path
    import pandas as pd

    paths = {subj: f"{scores_path}/{model_name}-{pt}-{dataset}-{subj}-{score_type}.npy" for subj in subjects}
    sdicts = {subj: np.load(paths[subj], allow_pickle=1).item() for subj in subjects} 

    # modeldict["scores"] # array of shape (n_layers, n_comps)
    # scores of shape [n_subj, n_layers, n_comps]
    
    agg_fn = np.mean#np.median
    
    scores = np.array([sdicts[subj]["scores"] for subj in subjects])
    #avg_scores= scores.mean(0);
    avg_scores= agg_fn(scores, axis=0);
    
    dxx=[]
    for subj in subjects:
        if not("mbwscores") in sdicts[subj].keys():
            print(f"Missing mbwscores at {model_name} {dataset} {subj}")
            continue;
        
        dxx.append(sdicts[subj]["mbwscores"])
           
    bwscores = np.array(dxx)      
    avg_bwscores= agg_fn(bwscores, axis=0);
    
    df = None;
    if do_df:
        df =pd.concat([make_df_(sdicts[subj], subj, layer_names=sdicts[subj]["layers"]) 
                  for subj in subjects])
    d = dict(scores=avg_scores, mbwscores=avg_bwscores,fulldf=df);
    return d;

"""
print(sdicts[subj].keys())
avg = average_scores_across_subjects(model_name, dataset, subjects)
avg.keys(), avg["scores"].shape
""";

##############################################################################
## Randomness, permuations and significance testing

def compute_block_perm_indices(num_timepoints, block_size=10):
    """
    Precompute block indices for block permutation.

    Parameters:
        num_timepoints (int): Total number of timepoints in the fMRI signal.
        block_size (int): Size of each block.

    Returns:
        indices (ndarray): Permuted indices for reordering timepoints.
    """
    # Calculate the number of full blocks and remainder size
    num_full_blocks = num_timepoints // block_size
    remainder_size = num_timepoints % block_size

    # Create indices for full blocks
    full_block_indices = np.arange(num_full_blocks * block_size).reshape(num_full_blocks, block_size)

    # Permute the order of full blocks
    np.random.shuffle(full_block_indices)
        
    permuted_indices = full_block_indices.flatten()

    # Handle the remainder block indices
    if remainder_size > 0:
        remainder_indices = np.arange(num_full_blocks * block_size, num_timepoints)
        permuted_indices = np.concatenate([permuted_indices, remainder_indices])

    return permuted_indices

def make_kfolds(num_points, k=5, block_size=10, perm_type="random"):
    idxs = np.arange(num_points)
    if perm_type.startswith("block"):
        idxs = compute_block_perm_indices(num_points, block_size=block_size);
    elif perm_type.startswith("rand"):
        idxs = np.random.permutation(num_points)
    
    idxs = idxs.astype(int)
    
    s = np.array_split(idxs, 5)
    folds =[]
    for i in range(k):
        train = []
        for a in s[:i]: train = train + list(a)
        for a in s[i+1:]: train = train + list(a)
        train = np.array(train)
        folds.append(( train , s[i]))
    
    return folds


"""
from lib.stats_and_metrics import compute_block_perm_indices, make_kfolds

a = np.arange(30)+10
folds = make_kfolds(30, k=5, perm_type="blocks")
for train, test in folds:
    print(test)
    result[test] = a[test]-10

# [10 11 12 13 14 15]
# [16 17 18 19 20 21]
# ....
result # [ 0.,  1.,  2.,  3 ...]

"""
##############################################################################
## New stats for significance testing

# // Testing for comparing two sets of predicted maps
# // e.g. by two separate models

from scipy import stats

# to ensure variables are normally distributed, which is required for t-test
def fisher_r_to_z(r):  
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_to_r(z):  
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# Perform the paired t-test across multiple variables (batch-wise)
def ttest_batch_test(model1_preds, model2_preds, **kwargs):
    # Apply t-test across all variables (columns) using np.apply_along_axis
    def ttest_for_variable(i):
        #from scipy.stats import wilcoxon: wilcoxon(model1_predictions, model2_predictions)
        return stats.ttest_1samp(model1_preds[:, i] - model2_preds[:, i], 0, **kwargs)
    
    # Vectorized t-test across columns (variables)
    results = np.apply_along_axis(ttest_for_variable, axis=0, arr=np.arange(model1_preds.shape[1]))
    return results


# // Permutation testing for single map

def split_into_chunks(array, chunk_size=10):
   return [array[i:i+chunk_size] for i in range(0, len(array), chunk_size)]

def get_permutated_indices(n_timepoints, nperm=5000, chunk_len=None):
    indices = np.arange(n_timepoints)
    permuted_indices_list = []
    chunks = indices[:, np.newaxis] if (chunk_len is None) else split_into_chunks(indices, chunk_len)
    for i in range(nperm):
        np.random.shuffle(chunks)
        permuted_indices_list.append(np.concatenate(chunks))

    return np.array(permuted_indices_list)

def perm_scores(preds, brain_data, idxperms=None, nperm=2000, chunk_len=10):
    n_rows, n_cols = brain_data.shape
    shape = (n_rows, n_cols)

    if idxperms is None:
        idxperms= get_permutated_indices(brain_data.shape[0], nperm, chunk_len)
    
    scores = []
    for perm_indices in tqdm(idxperms):
        res = corr_score(brain_data[perm_indices,:], preds)
        scores.append(res)

    # Compute p-values
    orig_score = corr_score(brain_data, preds)
    perm_scores = np.array(scores)
    pval = (perm_scores > orig_score).sum(axis=0) / nperm
    return orig_score, perm_scores, pval


## // Same as above, only parallelized across CPUs
import multiprocessing as mp
from multiprocessing import shared_memory, Value, Lock

def corr_score_view(brain_data, preds, perm_indices):
    # Use view for memory-efficient computation
    permuted_brain_data = brain_data[perm_indices, :]
    return corr_score(permuted_brain_data, preds)

def worker_compute_score(perm_indices, shm_name_brain, shm_name_preds, shape):
    # Attach to shared memory and restore shared arrays
    existing_brain = shared_memory.SharedMemory(name=shm_name_brain)
    existing_preds = shared_memory.SharedMemory(name=shm_name_preds)
    brain_data = np.ndarray(shape, dtype=np.float32, buffer=existing_brain.buf)
    preds = np.ndarray(shape, dtype=np.float32, buffer=existing_preds.buf)
    # Compute the score using view-based indexing
    score = corr_score_view(brain_data, preds, perm_indices)
    with lock: progress_counter.value += 1
    existing_brain.close(); existing_preds.close();
    return score

def init_worker(counter, lock_):
    global progress_counter, lock
    progress_counter = counter
    lock = lock_

def score_with_parallelization(preds, brain_data, idxperms=None, nperm=2000, chunk_len=10, use_tqdm=True):
    n_rows, n_cols = brain_data.shape
    shape = (n_rows, n_cols)

    if idxperms is None:
        print("Comp permutations")
        idxperms= get_permutated_indices(brain_data.shape[0], nperm, chunk_len)
        print("done.")

    # Create shared memory for large arrays and copy the data into shared memory buffers
    shm_brain = shared_memory.SharedMemory(create=True, size=brain_data.nbytes)
    shm_preds = shared_memory.SharedMemory(create=True, size=preds.nbytes)
    shm_brain_array = np.ndarray(shape, dtype=brain_data.dtype, buffer=shm_brain.buf)
    shm_preds_array = np.ndarray(shape, dtype=preds.dtype, buffer=shm_preds.buf)
    np.copyto(shm_brain_array, brain_data)
    np.copyto(shm_preds_array, preds)

    # Shared counter and lock for synchronized progress updates
    progress_counter = Value('i', 0)
    lock = Lock()

    # Setup the progress bar
    pbar = tqdm(total=nperm) if use_tqdm else None
    def progress_updater(results):
        if pbar:
            with lock:
                pbar.update(1)

    print("Start multiprocessing.")
    # Multiprocessing pool setup
    with mp.Pool(processes=16, initializer=init_worker, initargs=(progress_counter, lock)) as pool:
        async_results = []
        for perm_indices in idxperms:
            args = (perm_indices, shm_brain.name, shm_preds.name, shape)
            async_results.append(pool.apply_async(worker_compute_score, args, callback=progress_updater))

        # Wait for all tasks to complete
        scores = [result.get() for result in async_results]

    # Cleanup shared memory
    shm_brain.close(); shm_brain.unlink(); shm_preds.close(); shm_preds.unlink();

    if pbar: pbar.close()

    # Compute p-values
    orig_score = corr_score_view(brain_data, preds, np.arange(brain_data.shape[0]))
    perm_scores = np.array(scores)
    pval = (perm_scores > orig_score).sum(axis=0) / nperm
    return orig_score, perm_scores, pval


##############################################################################
## Memory stats etc (old) -> move to monitoring
"""
def list_slurm_info():
    for x in os.environ:
      if "slurm" in x.lower():
        print(x, os.environ[x])

def get_num_assigned_cpus():
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_ids= process.cpu_affinity()
    #SLURM_CPUS_ON_NODE 14, SLURM_JOB_NUM_NODES 1
    #SLURM_MEM_PER_NODE 29696, SLURM_NTASKS 14, SLURM_NPROCS 14
    # SLURM_NNODES 1, SLURM_JOB_CPUS_PER_NODE 14
    return len(cpu_ids);


def get_cuda_device_stats(device, n=0, rs = lambda x: x, udiv = (1024**3), unit="gb"):
  #cmem_all = torch.cuda.memory_allocated(device = device)/ unit
  cmem_all = torch.cuda.torch.cuda.memory_reserved(device = device)/ udiv
  cmem_max = torch.cuda.get_device_properties(device = device).total_memory/udiv;
  cmem_cached=torch.cuda.memory_cached(device = device)/ udiv
  #cmem_perc= torch.cuda.memory_usage(device=device)
  dname= torch.cuda.get_device_name(device=device);
  torch.cuda.synchronize()
  data = {f"cuda{n}": dname, f"cuda{n}_mem_res": rs(cmem_all)+unit, f"cuda{n}_mem_cached": rs(cmem_cached)+unit, \
          f"cuda{n}": rs(cmem_all*100/cmem_max)+"%", f"cuda{n}_mem_tot" : rs(cmem_max)+unit} #, "%cuda": cmem_perc
  return data;

def get_machine_stats(verb=False, gpu=False, ret=True, unit="gb", sep="\t", proc=False, rnd=False, per_cpu=False, per_gpu=False):
  unit2div = {"gb": (1024**3), "mb": (1024**2)};
  if not (unit in unit2div.keys()): unit = "gb";
  udiv = unit2div[unit];
  
  rs = lambda x : str(round(x, rnd)) if rnd != False else str(x);
  
  # CPU; average CPU utilization since last call (?)
  cpu_perc = psutil.cpu_percent()
  # Memory 
  vmem_perc = psutil.virtual_memory().percent;
  vmem_total = psutil.virtual_memory().total / udiv
  data = {"cpu": rs(cpu_perc)+"%", "vmem" : rs(vmem_perc) +"%", "vmem_tot" : rs(vmem_total)+unit , "n_cpus": psutil.cpu_count()}
  
  if proc:
    pid = os.getpid() if proc == True else proc;
    process = psutil.Process(pid)
    proc_mem = process.memory_info()[0]/ udiv; # (1024**3) ~ GB...I think
    proc_perc = proc_mem*100 / (vmem_total)
    data.update({"pid": pid, "proc_mem" : rs(proc_mem)+unit, "proc_mem/tot": rs(proc_perc)+"%" })
    cpu_ids= process.cpu_affinity()
    sum_cpu_perc = process.cpu_percent(); 
    data.update({"proc_n_cpus": len(cpu_ids), "proc_cum_cpu": rs(sum_cpu_perc)+"%", \
                 "proc_avg_cpu": rs(sum_cpu_perc/len(cpu_ids)) + "%"});
                 
    if per_cpu: 
        assigned_cpu_usage = np.array(psutil.cpu_percent(interval=None, percpu=True))[cpu_ids]
        data.update({"proc_assigned_cpus": cpu_ids, "assigned_cpu_util_%" : assigned_cpu_usage.round(rnd) })
        pass;
    
  # GPU memory usage whith pytorch
  if gpu!=False:
   if not("torch" in sys.modules):  
        print("please import torch first")
   
   if not torch.cuda.is_available():
     data.update({"cuda": "not_avail"});
    
   elif per_gpu: 
    for x in range(torch.cuda.device_count()):
        device = torch.device("cuda:"+ str(x));
        data.update(get_cuda_device_stats(device, x, rs, udiv, unit));
   else:
    device = torch.device("cuda" if gpu==True else gpu) 
    data.update(get_cuda_device_stats(device, "", rs, udiv, unit));
    
  if verb:
    print(sep.join([f"{k}: {data[k]}" for k in data.keys()]))
  if ret: 
    return data;



import sys


def deepsizeof(obj):
   if isinstance(obj, dict):
     return np.sum([deepsizeof(v) for k,v in  obj.items()])
   if isinstance(obj, (np.ndarray, np.generic) + (() if not ("torch" in globals().keys()) is None else (torch.Tensor)) ):
     return obj.nbytes;
   if isinstance(obj, list):
     return np.sum([deepsizeof(v) for v in  obj]);
   return sys.getsizeof(obj)
   
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def list_biggest_vars(namespace , n=10): # namespace e.g. = globals() or locals()
  for name, size in sorted(((name, deepsizeof(value)) for name, value in list(namespace.items())), key= lambda x: -x[1])[:n]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def tqdm_mem_stats(per_cpu=False):
    # [k for k in os.environ.keys() if "SLURM" in k]
    r = get_machine_stats(verb=0,gpu=0, ret=1, sep="  ", proc=1, rnd=2, per_cpu=1)
    cpup=r["assigned_cpu_util_%"].mean().round(2)
    if per_cpu:
        print(r["assigned_cpu_util_%"])
    pm = r["proc_mem"]
    alloc_ram = str(int(int(os.environ["SLURM_MEM_PER_NODE"]) / 1024)) + "gb";
    free = round(float(r["vmem_tot"][:-2]) * (1-(float(r['vmem'][:-1])/100)))
    s = f"c:{round(cpup)}% r:{round(float(pm[:-2]))}/{alloc_ram} f:{free}gb"
    return s;


def select_x_items(full_list, x=5):
    if x > len(full_list):
        return full_list;
    idx = np.round(np.linspace(0, len(full_list) - 1, x)).astype(int)
    return list(np.array(full_list)[idx])
"""


