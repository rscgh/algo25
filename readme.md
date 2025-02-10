
# Contents


## Repository structure

* **brainannlib** is a very lightweight pacakge containing shared functions
* **scripts** - reliaze one specific step (e.g. collecting activations of an ANN to stimuli and saving them) by making use of the common functions defined in brainannlib, usually to be run on slurm nodes
* **notebooks** - visualization of results and lightweight analysis computation

## Installation 

```sh
# clone the github directory
git clone ...

# create a conda environment with the necessary dependencies (~4GB)
# possibly you have to first load the module, e.g by module load conda|anaconda3|miniconda
# for more troubleshooting see the paragraph below
conda env create -f environment_conda_algoenv.yml --solver=libmamba
# once we expand to more networks, this will likely have to be updated in the yml file

# activate the environment
conda activate algoenv

# [optional:] install the python kernel so you can use the environment from within jupyterhub
python -m ipykernel install --user --name algoenv --display-name "algoenv"

# go th the brainlib folder and install it as a package
# to later enable imports like
# brainannlib.anns import load_model
# can be uninstalled later using: pip uninstall brainannlib
cd highlv_ann/brainannlib
pip install -e .

# Lastly set up the needed paths to the datasets. Please change the paths to suit your local system.
# You can even use the same directory for this
export ALGONAUTS_ROOT_DIR="/scratch-scc/users/robert.scholz2/cneuromod"
# per-episode ANN activations
mkdir -p $ALGONAUTS_ROOT_DIR/ann_brain_data/activations
# cummulated+reduced activations, saved regression models and predictions
mkdir -p $ALGONAUTS_ROOT_DIR/ann_brain_data/outputs

# To avoid having to rerun it everytime you start a new shell, you can add it also to your user profile
echo 'export ALGONAUTS_ROOT_DIR="/scratch-scc/users/robert.scholz2/cneuromod"' >> ~/.profile
```

**Notes on conda env create**
* this will install all the required packages (using conda if available, and otherwise pip)
* not all packages may be required, depending on which ANNs you use 
* once we test more networks, we might have to update this
* also first check if the right target version of cuda is given, you can check what cuda version your node supports by running nvidia-smi in the shell (possibly you will have to 'module load cuda' or similiar first)
* The classic solver from conda is likely too slow to install all the packages. From conda 23.10, the faster conda-libmamba-solver is the default solver. For earlier versions, if possible either update conda or install the solver; if you cannot do that (e.g. due to access restrictions in a server environment) you can try to use dropin replacements such as mamba or micromamba, e.g. `module load micromamba` and `micromamba create --file environment_conda_algoenv.yml`. The environments should be fully compatible with conda
`conda config --add envs_dirs /mnt/vast-standard/home/robert.scholz2/u14262/micromamba/envs`


## Download of the challenge files
datalad is like git with big file support

```sh
# go to the project root folder
cd $ALGONAUTS_ROOT_DIR

# the following command clones the git repo to a local dir "algonauts_2025.competitors" (without downloading the files)
datalad install -r git@github.com:courtois-neuromod/algonauts_2025.competitors.git

# donwload all the data (with r for recusrively going in subfolders and J8 for 8 parallel jobs)
# this will download ~ 2.3G of fmri-data and 109G stimuli
datalad get -r -J8 .

# or download only subdirectories
datalad get -r -J8 fmri/*
datalad get -r -J8 stimuli/*
```


## Running 

For examples, check the scripts under /scripts/


## Testing