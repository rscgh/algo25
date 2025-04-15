import time
import psutil
import os
import numpy as np

def init_cpu_monitoring(current_user=None, pid=None):

    if current_user is None:
        current_user = os.environ.get("USER") or os.environ.get("USERNAME")

    cpu_ids=[]
    user_processes = get_user_processes(current_user) if pid is None else [psutil.Process(pid)]
    for proc in user_processes:
        
        #if not("cpu_affinity" in proc.info.keys()):continue;
        cpu_ids=cpu_ids+proc.cpu_affinity();

    cpu_ids = np.unique(cpu_ids)
    if len(cpu_ids)==0: cpu_ids=slice(None)
    return cpu_ids

def get_user_processes(current_user=None, excl_system=False):
    excludes=["slurm_script", "systemd", "(sd-pam)", "starter", "squashfuse_ll", "sshd"]

    if current_user is None:
        current_user = os.environ.get("USER") or os.environ.get("USERNAME")
    
    procs=[p for p in psutil.process_iter(['pid', 'username', 'memory_info']) if p.info['username'] == current_user];
    if excl_system: procs=[p for p in procs if not(p.name() in excludes)]
    return procs


def list_processes(current_user=None, full=False, show_affinity=False):
    this_pid = os.getpid()
    user_processes = get_user_processes(current_user)
    for p in user_processes:

        print(p.pid, p.name(), " "*(25-len(p.name())), round(p.info['memory_info'].rss / (1024**3),2),"gb  ", end="")
        if full: print(p.info, end="");
        if show_affinity: print("\t",len(p.cpu_affinity()), p.cpu_affinity(), end="");
        if p.pid==this_pid: print("\033[1m<-- current\033[0m", end="") 
        print("")



import os, time, psutil, numpy as np
from brainannlib.monitoring import get_user_processes

def monitor(period_s=60, sleep_s=2, cpu_ids=None, pids=None, ipython_reset=False, 
            show_indiv=False, show_per_process=False, cuda=False, disk=False):
    if ipython_reset:
        from IPython.display import clear_output  

        current_user = os.environ.get("USER") or os.environ.get("USERNAME")  
    pids = [pids] if isinstance(pids, int) else pids  

    if cpu_ids is None:
        cpu_ids = set()  
        if pids is None:
            cpu_ids.update(init_cpu_monitoring(current_user, pid=pids))  
        else:
            for pid in pids: cpu_ids.update(init_cpu_monitoring(current_user, pid=pid))  
        
        cpu_ids = sorted(cpu_ids)  

    if cuda:
        import torch
        for dev in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats()  
    
    imax = period_s // sleep_s  
    peak_ram, cumm_reads, cumm_writes = 0, {}, {}  
    try:
        for i in range(1, imax + 1):
            if ipython_reset: clear_output(wait=True)  

            processes = [psutil.Process(pid) for pid in pids] if pids else get_user_processes(current_user, excl_system=True)  
            total_ram_used = sum(p.memory_info().rss for p in processes) / (1024**3)  
            
            # use slurm assigned node memory (more specific to the user) if availabe
            # otherwise use just the entire memory available on the node as refenrence
            slurm_mem_per_node =os.getenv("SLURM_MEM_PER_NODE")
            total_ram_avail = psutil.virtual_memory().total / (1024**3) \
                if slurm_mem_per_node is None else int(slurm_mem_per_node) / (1024) 
            
            peak_ram = max(peak_ram, total_ram_used)  
            mem_pct = (total_ram_used / total_ram_avail * 100);
            print(f"RAM: \033[1m{total_ram_used:.3f} GB\033[0m {mem_pct:.1f}%, step={i}/{imax} (peak: {peak_ram:.3f} GB)")  

            if show_per_process:
                for p in processes:
                    ram = p.memory_info().rss / (1024**3)  
                    mem_pct = (ram / total_ram_avail * 100);
                    name = p.name()  

                    if disk:
                        io = p.io_counters()
                        prev_read = cumm_reads.get(p.pid, io.read_bytes)
                        prev_write = cumm_writes.get(p.pid, io.write_bytes)
                        read_rate = (io.read_bytes - prev_read) / sleep_s / (1024**2)  
                        write_rate = (io.write_bytes - prev_write) / sleep_s / (1024**2)  
                        cumm_reads[p.pid], cumm_writes[p.pid] = io.read_bytes, io.write_bytes  
                        io_str = f"{read_rate:.2f}mb/s {write_rate:.2f}mb/s"  
                    else:
                        io_str = ""  

                    print(f"{p.pid} {name}\t {ram:.3f}GB\t{mem_pct:.1f}%\t  {io_str}".strip())  

            cpu_util = np.array(psutil.cpu_percent(percpu=True))[cpu_ids]  
            print(f"CPUs (n={len(cpu_ids)}): \033[1mavg={cpu_util.mean():.1f}%\033[0m", end="")  
            if show_indiv:
                print(" | non-idle: " + " ".join(f"{v:.1f}%" for v in cpu_util[cpu_util.round().astype(int) != 0]), end="")  
            print()  

            if cuda:
                for dev in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(dev)  
                    mem = torch.cuda.memory_allocated(dev) / 1024**3  
                    peak_mem = torch.cuda.max_memory_allocated(dev) / 1024**3  
                    print(f"GPU {props.name}: \033[1m{mem:.2f}\033[0m/{props.total_memory / 1024**3:.2f}GB (Peak: {peak_mem:.2f}GB)")  

            time.sleep(sleep_s)  


    except KeyboardInterrupt:
        print("Interrupted")  

    print(f"Max memory usage over {period_s}s: {peak_ram:.3f}GB")  


"""
def monitor(period_s = 60, sleep_s=2, cpu_ids=None, pid=None, \
            ipython_reset=False, current_user=None, \
            show_indiv=False, cuda=False, disk=False):
    i=0;
    if ipython_reset:
        from IPython.display import clear_output
    
    if current_user is None:
        current_user = os.environ.get("USER") or os.environ.get("USERNAME")

    if cpu_ids is None:
        cpu_ids= init_cpu_monitoring(current_user, pid=pid);
    
    if cuda:
        import torch
        for dev in range(torch.cuda.device_count()):
            print(dev)
            device = torch.device(f"cuda:{dev}")
            print(device)
            print(torch.cuda.is_available())  # Should return True
            torch.cuda.reset_peak_memory_stats();

    peak_ram = 0;    
    cumm_reads, cumm_writes=0,0

    imax=int(period_s/sleep_s)

    try:
        while i+1<=imax:
            if ipython_reset: clear_output(wait=True);
            user_processes = get_user_processes(current_user, excl_system=True) if (pid is None) else [psutil.Process(pid)]
            # RAM
            total_ram_usage = sum(proc.memory_info().rss for proc in user_processes) / (1024**3)
            if total_ram_usage > peak_ram: peak_ram = total_ram_usage;
            print(f"RAM: \033[1m {total_ram_usage:.3f} GB\033[0m, step={i+1}/{imax} ... with peak at {peak_ram:.3f}GB")
            
            # Disk
            if disk:
                read = sum(p.io_counters().read_bytes for p in user_processes)
                write = sum(p.io_counters().write_bytes for p in user_processes)
                reads_per_s = round(((read-cumm_reads) / period_s)/(1024**2),2)
                writes_per_s = round(((write-cumm_writes) / period_s)/(1024**2),2)
                cumm_reads=read; cumm_writes=write;
                print(f"Disk IO: \033[1mread={reads_per_s}mb/s, write={writes_per_s}mb/s\033[0m")

            # CPU
            cpu_per_core = psutil.cpu_percent(percpu=True)
            util=np.array(cpu_per_core)[cpu_ids]; 
            nonzero=util[np.round(util).astype(int)!=0]
            ins="" if isinstance(cpu_ids, slice) else f" ({len(cpu_ids)} assinged)"
            print(f"CPUs{ins}: \033[1mavg={np.mean(util).round(1)}%\033[0m")
            if show_indiv: print("... individual assigned non-idle:\n ", nonzero);

            # GPU
            if cuda:
                for dev in range(torch.cuda.device_count()):
                    name=torch.cuda.get_device_name(dev)
                    curr=torch.cuda.memory_allocated(dev)
                    max_alloc = torch.cuda.max_memory_allocated(dev) 
                    max_mem = torch.cuda.get_device_properties(dev).total_memory
                    print(f"GPU {name}: \033[1m{curr / 1024**3:.2f}\033[0m/{max_mem / 1024**3:.2f}gb ... (Peak: {max_alloc / 1024**3:.2f}gb)")

            time.sleep(sleep_s)
            i=i+1
    except KeyboardInterrupt:
        print("Interruped")
    
    print(f"Max memory usage over the last {period_s}s: {peak_ram:.3f}GB")
"""

def get_num_assigned_cpus():
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_ids= process.cpu_affinity()
    #SLURM_CPUS_ON_NODE 14, SLURM_JOB_NUM_NODES 1
    #SLURM_MEM_PER_NODE 29696, SLURM_NTASKS 14, SLURM_NPROCS 14
    # SLURM_NNODES 1, SLURM_JOB_CPUS_PER_NODE 14
    return len(cpu_ids);


###########################################################################################################
## Freeing shared multiprocessing memory (optional)

import psutil
import os

import subprocess
import re
import os

def get_shm_id_from_key(shm_key):
    # Run 'ipcs -m' to list shared memory segments
    result = subprocess.run(['ipcs', '-m'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise Exception("Error running ipcs -m")
    
    # Parse the result for the key associated with 'psm_*'
    for line in result.stdout.splitlines():
        # Each line contains segments like: 
        # "0x12345678   12345  user  group  ..."
        # We need to extract the key and segment ID
        match = re.match(r"\s*(\d+)\s+(\S+)\s+.*", line)
        if match:
            shm_id = match.group(1)  # Segment ID
            key = match.group(2)     # Key in hexadecimal form
            
            # If the key matches, return the segment ID
            if key == shm_key:
                return shm_id
    return None

def cleanup_shared_memory_key(shm_key):
    # Get the shared memory ID from the key
    shm_id = get_shm_id_from_key(shm_key)
    
    if shm_id is not None:
        print(f"Found shared memory with ID {shm_id}. Cleaning up...")
        # Use ipcrm to clean up shared memory by ID
        os.system(f"ipcrm -m {shm_id}")
        print(f"Shared memory with ID {shm_id} cleaned up.")
    else:
        print(f"No shared memory found for key {shm_key}.")

# Example usage
#shm_key = "psm_ceb72c0f"  # This should be the name you're working with
#leanup_shared_memory_key(shm_key)

def cleanup_shared_memory():
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if proc.info['name'] == 'python':  # Look for Python processes
            try:
                # Check if the process has shared memory segments
                for mmap in proc.memory_maps():
                    if 'shm' in mmap.path:  # Filter out shared memory segments
                        print(f"Found orphaned shared memory: {mmap.path}")
                        # You can attempt to remove it with ipcrm here if you find the key
                        # Extracting the segment ID/key
                        shm_key = mmap.path.split('/')[-1]  # This may contain the key or ID
                        if shm_key.startswith('psm_'):
                            print(f"Trying to clean up shared memory: {shm_key}")
                            # Clean using os.system for key-based removal
                            cleanup_shared_memory_key(shm_key)
                            #os.system(f"ipcrm -M {shm_key}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


import sys
import numpy as np

def list_global_vars(min_size=1e6):
    """
    Lists global variables that are larger than the specified size.
    If the object is a numpy array, also displays its shape.
    
    Parameters:
    min_size (float): The minimum size (in bytes) of the objects to be displayed.
                      Default is 1MB (1e6 bytes).
    """
    # Get all objects in memory from globals
    objects_with_size = []

    # Iterate through global variables
    for name, obj in globals().items():
        size = sys.getsizeof(obj)
        
        if size > min_size:  # Only show objects greater than min_size
            # If it's a numpy array, also show its shape
            shape = getattr(obj, 'shape', 'N/A') if isinstance(obj, np.ndarray) else 'N/A'
            objects_with_size.append((name, type(obj), size, shape))

    # Sort the objects by size in descending order
    sorted_objects = sorted(objects_with_size, key=lambda x: x[2], reverse=True)

    # Print the objects and their sizes
    for name, obj_type, size, shape in sorted_objects:
        print(f"Object Name: {name}, Type: {obj_type}, Size: {size / 1e6:.2f} MB, Shape: {shape}")




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

