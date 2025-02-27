
# Contents


## Repository structure

* **brainannlib** is a very lightweight pacakge containing shared functions
* **scripts** - reliaze one specific step (e.g. collecting activations of an ANN to stimuli and saving them) by making use of the common functions defined in brainannlib, usually to be run on slurm nodes
* **notebooks** - visualization of results and lightweight analysis computation

## Installation 

```sh
# clone the github directory
git clone git@github.com:rscgh/algo25.git

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
cd algo25 #/brainannlib
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

An example pipeline (leading to the current submission results) looks like the following:
```sh
python -u scripts/collect_smollm2_activations.py
python -u scripts/reduce_smollm2_activations.py
# repeat these two steps for the other modalities [...]
python -u scripts/regress_combined.py
```


## Resource requirements

```sh
# for the combined regression model fitting (min 40gb RAM, so far uses only one CPU)
srun --time=3:00:00 --export=ALL --partition=scc-cpu --ntasks=1 --nodes=1 --cpus-per-task=2 --mem=40G --pty bash

# for collecting activations, CPUs could be less
# and possibly the memory too, as whats mostly matters is the GPU memory
srun --time=3:00:00 --export=ALL --partition=scc-gpu --gres=gpu:1 --ntasks=1 --nodes=1 --cpus-per-task=12 --mem=40G --pty bash

```

## Working with the git repo

**Working on a your own (feature) branch**
```sh
git checkout -b feature/my-feature  # -b creates

# within the feature branch, commit changes
git add .
git commit -m "Description of changes"
```

To make it clearer what a specific branch does, we can stick to branch naming conventions such as
```sh
feature/language-model-llamav32         # e.g. including feature reduction
feature/vision-models-various           # e.g. including multiple models
feature/looped-feature-selection        # including on specific feature selection method
release/submission-v2.0                 # whenever we try to work on a new submission
# If you can think of other branch-types that could be useful we can add them here.
```

**Updating and pushing of the branch**
```sh
# change to main branch again & ensure the local main is up-to-date with the remote
git checkout main
git pull origin main  

# include all the updates from the main
# in the branch through rebasing
git checkout feature/my-feature
git rebase main

# in case of conflict, edit the files (resolving conflicts)
# and then continue the rebase
git add <file-with-conflict>
git rebase --continue
# or alternatively abort: git rebase --abort

# push the rebased branch to the remote
# need to force, because rebase might have
# has changed # the commit history
git push origin feature/my-feature --force
```

**Integrating branch into main through pull request**
```sh
# Go to GitHub → your repository → open a pull request from my-feature-branch to main.
# After approval, merge the PR on GitHub.

# On your machine, pull these changes
git checkout main
git pull origin main
# optionally delete the branch:
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```